
import os
import copy
import utils
import time
import json
import numpy as np
from sentence_transformers import util

from model_interface_emb import SglModelAsyncEmb

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



models_dict = dict()


# models_dict['gpt-oss-120b'] = get_model(model="v_llm/gpt-oss-120b", base_url="https://pn131285.nist.gov:8447/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))


# models_dict['openai/gpt-oss-20b'] = "iarpa018:8443"
# models_dict['Qwen/Qwen3-30B-A3B-Instruct-2507'] = "iarpa017:8444"

models_dict['meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'] = "pn131275:8443"
models_dict['openai/gpt-oss-120b'] = "pn131285:8444"






# base_dir = './data-subset-500-emb'
base_dir = './data-post-cutoff-emb'

preamble = """Answer the following open ended short answer question. The last line of your response should be of the following format: 'ANSWER: $answer' (without quotes) where answer is the answer to the question. Think step by step before answering."""



available_models = list(models_dict.keys())

for model_name in available_models:
    print(80*'=')
    print(model_name)
    print(80*'=')
    remote = models_dict[model_name]
    model = SglModelAsyncEmb(remote=remote, model=model_name, connection_parallelism=128)

    for ds in ['oe-Q235B-filtered', 'oe-gpt120b-filtered']:
        dataset_fldr = os.path.join(base_dir, ds)
        if not os.path.exists(dataset_fldr):
            continue

        json_files = [fn for fn in os.listdir(dataset_fldr) if fn.endswith('.json')]
        for json_fn in json_files:
            cur_d_fp = os.path.join(dataset_fldr, json_fn)
            mn = model_name.split('/')[-1]

            with open(cur_d_fp, 'r') as f:
                data = json.load(f)

            valid = True
            for sample in data:
                if not f'{mn}_embeddings' in sample.keys():
                    valid = False
                    break
                if not 'cosine_embO_embOwC' in sample[f'{mn}_embeddings'].keys() or not 'cosine_embR_embOwC' in sample[f'{mn}_embeddings'].keys():
                    valid = False
                    break
            if valid:
                print("--------------------------------")
                print(f"Skipping (already copmleted) {cur_d_fp} embeddings with {mn}")
                continue


            print("--------------------------------")
            print(f"Processing {cur_d_fp} embeddings with {mn}")

            


            prompts = []
            for sample in data:
                q = f"{preamble}\n\n{sample['orig_question']}"
                prompts.append(q)
            results, total_time = model.generate(prompts)
            for i, res in enumerate(results):
                sample = data[i]
                sample['orig_question_embedding'] = (res['embedding'])


            prompts = []
            for sample in data:
                q = f"{preamble}\n\n{sample['question']}"
                prompts.append(q)
            results, total_time = model.generate(prompts)
            for i, res in enumerate(results):
                sample = data[i]
                sample['question_embedding'] = (res['embedding'])


            prompts = []
            for sample in data:
                q = f"{preamble}\n\n<context>{sample['context']}</context>\n\n{sample['orig_question']}"
                prompts.append(q)
            results, total_time = model.generate(prompts)
            for i, res in enumerate(results):
                sample = data[i]
                sample['orig_question_w_context_embedding'] = (res['embedding'])

            orig_question_w_context_embedding = np.concatenate([np.asarray(s['orig_question_w_context_embedding']).reshape(1, -1) for s in data], axis=0)
            question_embedding = np.concatenate([np.asarray(s['question_embedding']).reshape(1, -1) for s in data], axis=0)
            orig_question_embedding = np.concatenate([np.asarray(s['orig_question_embedding']).reshape(1, -1) for s in data], axis=0)

            similarities_embO_embOwC = util.pytorch_cos_sim(orig_question_embedding, orig_question_w_context_embedding).cpu().numpy()
            similarities_embR_embOwC = util.pytorch_cos_sim(question_embedding, orig_question_w_context_embedding).cpu().numpy()

            
            for i, sample in enumerate(data):    
                sample[f'{mn}_embeddings'] = dict()
                sample[f'{mn}_embeddings']['cosine_embO_embOwC'] = similarities_embO_embOwC[i,i]
                sample[f'{mn}_embeddings']['cosine_embR_embOwC'] = similarities_embR_embOwC[i,i]
                del sample['orig_question_w_context_embedding']
                del sample['orig_question_embedding']
                del sample['question_embedding']

            avg_embO_embOwC = np.mean([s[f'{mn}_embeddings']['cosine_embO_embOwC'] for s in data])
            avg_embR_embOwC = np.mean([s[f'{mn}_embeddings']['cosine_embR_embOwC'] for s in data])
            print(f"avg_embO_embOwC = {avg_embO_embOwC}")
            print(f"avg_embR_embOwC = {avg_embR_embOwC}")

            
            with open(cur_d_fp, 'w') as f:
                json.dump(data, f, indent=2)


    
    