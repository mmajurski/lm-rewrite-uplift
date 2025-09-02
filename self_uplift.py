import numpy as np
import os
import random
import json
import time
import random

import answer_parser
from model_interface import SglModelAsync
import utils
import copy

import prompts




def reformat_questions(dataset: list[dict], remote:str, model:str, reasoning_effort:str='high', connection_parallelism:int=64) -> list[dict]:

    # build the prompts
    q_prompts = [prompts.QUESTION_SELF_UPLIFT_PROMPT.format(question=d['orig_question']) for d in dataset]

    model = SglModelAsync(remote=remote, model=model, reasoning_effort=reasoning_effort, connection_parallelism=connection_parallelism)

    results, total_time = model.generate(q_prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    to_delete = list()
    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        else:
            dataset[i]['reformat_response'] = res['content']
            dataset[i]['reformat_scratchpad'] = res['scratchpad']
            parsed = answer_parser.parse_question_open(res['content'])
            if parsed is None:
                to_delete.append(i)
                # raise Exception(f"Failed to parse response: {res['content']}")
            else:
                dataset[i]['question'] = parsed['question']

    dataset = copy.deepcopy(dataset)
    for i in sorted(to_delete, reverse=True):
        del dataset[i]

    return dataset
        

    
    
    



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts a jsonl dataset into MMLU format to be used as an MCQ evaluation.')
    parser.add_argument('--sample_count', type=int, default=-1, help='number of samples to generate, set to <=0 for all')
    parser.add_argument('--dataset', type=str, default='squadv2.jsonl', help='dataset to generate, options: squadv2, ucinlp_drop')
    parser.add_argument('--src_dataset_dir', type=str, default='./data-subset-500', help='source dataset directory')
    parser.add_argument('--out_dataset_dir', type=str, default='./data-subset-500-out', help='output dataset directory')
    parser.add_argument('--remote', type=str, default="sierra")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument('--reasoning_effort', type=str, default='high', help='reasoning effort, options: high, medium, low')
    parser.add_argument('--connection_parallelism', type=int, default=256, help='connection parallelism, set low for gpt-oss to try and avoid Harmony errors')

    args = parser.parse_args()
    # print("Generating reformatted questions")
    # args.dataset = "flashrag_2wikimultihopqa.json"
    # args.src_dataset_dir = "./data-subset-1000/oe-unmodified-subset/"
    # args.out_dataset_dir = "./data-subset-1000/oe-gpt120b-tmp"
    # args.remote = "pn131285:8447"
    # args.model = "gpt-oss-120b"
    # args.reasoning_effort = 'high'
    # args.out_dataset_dir = "./data-subset-1000/oe-Q235B"
    # args.remote = "pn131285:8446"
    # args.model = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    

    base_path = os.path.splitext(args.dataset)[0]  # Remove extension
    fn_basename = os.path.basename(base_path)# + "_reformat"
    os.makedirs(args.out_dataset_dir, exist_ok=True)
    output_fn = os.path.join(args.out_dataset_dir, f'{fn_basename}.json')

    if os.path.exists(output_fn):
        print(f"Output file {output_fn} already exists, skipping")
        exit()

    print(args)

    start_time = time.time()
    args.dataset = os.path.join(args.src_dataset_dir, args.dataset)

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    if args.sample_count > 0 and args.sample_count < len(dataset):
        dataset = random.sample(dataset, args.sample_count)

    # verify that each element in the dataset has the following keys: question, answer, context
    for item in dataset:
        if 'question' not in item or 'answer' not in item or 'context' not in item:
            raise ValueError('each element in the dataset must have the following keys: question, answer, context')

    print("Dataset has %d contexts" % len(dataset))

    # Create a list to store model responses
    model_responses = list()
    # Copy over question (into orig_question), id, and context
    for item in dataset:
        response_item = {}
        if 'question' in item:
            response_item['orig_question'] = item['question']
        else:
            raise ValueError("No question found in {item}")
        if 'answer' in item:
            response_item['orig_answer'] = item['answer']
        else:
            raise ValueError("No answer found in {item}")
        if 'context' in item:
            response_item['context'] = item['context']
        else:
            raise ValueError("No context found in {item}")
        model_responses.append(response_item)
    dataset = model_responses

    
    if not os.path.exists(output_fn):
        model_dataset = reformat_questions(dataset, args.remote, args.model, args.reasoning_effort, args.connection_parallelism)

        print(f"Saving (N={n}) {len(model_dataset)} questions to {output_fn}")
        
        with open(output_fn, 'w') as f:
            json.dump(model_dataset, f, indent=2)



