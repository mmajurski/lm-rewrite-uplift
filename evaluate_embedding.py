
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

models_dict['Qwen/Qwen3-Embedding-8B'] = "pn120393:8443"
models_dict['intfloat/e5-mistral-7b-instruct'] = "pn120393:8445"
# models_dict['google/gemma-3-27b-it'] = "pn125915:8443"






def eval_embeddings(base_dir: str):
    available_models = list(models_dict.keys())

    for model_name in available_models:
        print(80*'=')
        print(model_name)
        print(80*'=')
        remote = models_dict[model_name]
        model = SglModelAsyncEmb(remote=remote, model=model_name, connection_parallelism=128)

        for ds in ['oe-Q235B-filtered', 'oe-gpt120b-filtered', 'oe-gpt20b-filtered', 'oe-Q235B-afc-filtered', 'oe-gpt120b-afc-filtered', 'oe-gpt20b-afc-filtered']:
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
                    ks = sample[f'{mn}_embeddings'].keys()
                    eval_keys = ['cosine_embR_embC', 'cosine_embO_embC']
                    for k in eval_keys:
                        if not k in ks:
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
                    prompts.append(sample['orig_question'])
                results, total_time = model.generate(prompts)
                for i, res in enumerate(results):
                    sample = data[i]
                    sample['orig_question_embedding'] = (res['embedding'])

                prompts = []
                for sample in data:
                    prompts.append(sample['reformat_question'])
                results, total_time = model.generate(prompts)
                for i, res in enumerate(results):
                    sample = data[i]
                    sample['question_embedding'] = (res['embedding'])

                prompts = []
                for sample in data:
                    prompts.append(sample['context'])
                results, total_time = model.generate(prompts)
                for i, res in enumerate(results):
                    sample = data[i]
                    sample['context_embedding'] = (res['embedding'])

                

                
                question_embedding = np.concatenate([np.asarray(s['question_embedding']).reshape(1, -1) for s in data], axis=0)
                orig_question_embedding = np.concatenate([np.asarray(s['orig_question_embedding']).reshape(1, -1) for s in data], axis=0)
                context_embedding = np.concatenate([np.asarray(s['context_embedding']).reshape(1, -1) for s in data], axis=0)

                similarities_embR_embC = util.pytorch_cos_sim(question_embedding, context_embedding).cpu().numpy()
                similarities_embO_embC = util.pytorch_cos_sim(orig_question_embedding, context_embedding).cpu().numpy()
                
                for i, sample in enumerate(data):    
                    sample[f'{mn}_embeddings'] = dict()
                    sample[f'{mn}_embeddings']['cosine_embO_embC'] = similarities_embO_embC[i,i]
                    sample[f'{mn}_embeddings']['cosine_embR_embC'] = similarities_embR_embC[i,i]

                    del sample['context_embedding']
                    del sample['orig_question_embedding']
                    del sample['question_embedding']

                avg_embO_embC = np.mean([s[f'{mn}_embeddings']['cosine_embO_embC'] for s in data])
                avg_embR_embC = np.mean([s[f'{mn}_embeddings']['cosine_embR_embC'] for s in data])
                print(f"avg_embO_embC = {avg_embO_embC}")
                print(f"avg_embR_embC = {avg_embR_embC}")
                


                
                with open(cur_d_fp, 'w') as f:
                    json.dump(data, f, indent=2)


    
    

if __name__ == '__main__':

    eval_embeddings('./data-subset-500')
    eval_embeddings('./data-post-cutoff')
    eval_embeddings('./data-subset-500-SU')
    eval_embeddings('./data-subset-500-afc')
    eval_embeddings('./data-post-cutoff-afc')