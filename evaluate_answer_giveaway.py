import numpy as np
import json
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answer_parser
from model_interface import SglModelAsync
import prompts


def get_key_prefix(reformat: bool):
    if reformat:
        return 'reformat_'
    else:
        return 'orig_'
    

def compute_meta_scores(dataset_fp, remote, model, reformat:bool):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)
    
    
    print(f"Computing meta properties for {dataset_fp}")
    print("Dataset has %d contexts" % len(dataset))

    # build the prompts
    if reformat:
        model_prompts = [prompts.ANSWER_GIVEAWAY_PROMPT.format(context=d.get('context', ''), question=d['question'], answer=d['orig_answer']) for d in dataset]
    else:
        model_prompts = [prompts.ANSWER_GIVEAWAY_PROMPT.format(context=d.get('context', ''), question=d['orig_question'], answer=d['orig_answer']) for d in dataset]
    

    model = SglModelAsync(remote=remote, model=model, connection_parallelism=64)
    results, total_time = model.generate(model_prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")

    
    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        else:
            parsed = answer_parser.parse_answer_giveaway_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
            else:
                dataset[i][f'{get_key_prefix(reformat)}answer_giveaway_score'] = parsed['answer_giveaway_score']
                dataset[i][f'{get_key_prefix(reformat)}answer_giveaway_response'] = res['content']
                dataset[i][f'{get_key_prefix(reformat)}answer_giveaway_scratchpad'] = res['scratchpad']


    scores = [d[f'{get_key_prefix(reformat)}answer_giveaway_score'] for d in dataset]
    print(f"Average answer giveaway score: {np.mean(scores)}")
    for s_thres in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(f"Number of questions with answer giveaway score <= {s_thres}: {np.sum(np.asarray(scores) <= s_thres)}")
    

    print(f"Saving {len(dataset)} questions to {dataset_fp}")
    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)



def evaluate_dataset_answer_giveaway_features(ifp, remote: str, model: str, reformat:bool):

    fns = []
    for fn in os.listdir(ifp):
        if fn.endswith('.json'):
            fns.append(fn)
    fns.sort()
    fns = fns[:1]
    if len(fns) == 0:
        print(f"evaluate_dataset_answer_giveaway_features: No datasets found in {ifp}")
        return
    print(f"evaluate_dataset_answer_giveaway_features: Found {len(fns)} datasets")
    print(f"evaluate_dataset_answer_giveaway_features: Datasets: {fns}")



    print(f"evaluate_dataset_answer_giveaway_features: Computing squishy statistics for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")
        compute_meta_scores(dataset_fp, remote, model, reformat)
    






if __name__ == '__main__':
    ifp = './data-subset-200/oe-Q235B-with-meta-properties/'
    # remote = 'pn131285:8446'
    # model = 'Qwen/Qwen3-235B-A22B-Instruct-2507-FP8'


    # ifp = './data-subset-200/oe-gpt120b-with-meta-properties/'
    remote = 'pn131285:8447'
    model = 'gpt-oss-120b'

    print(f"Evaluating answer giveaway features for {model} on the reformatted questions")
    evaluate_dataset_answer_giveaway_features(ifp, remote, model, reformat=True)
    print(f"Evaluating answer giveaway features for {model} on the original questions")
    evaluate_dataset_answer_giveaway_features(ifp, remote, model, reformat=False)