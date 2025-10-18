import numpy as np
import os
import random
import json
import time
import random

import answer_parser
from model_interface import SglModelAsync
import utils

import prompts


def build_og_msg(context:str, question:str, answer:str):
    
    prompt = prompts.QUESTION_REFORMAT_PROMPT.format(context=context, question=question, answer=answer)
    return prompt


def reformat_questions(dataset: list[dict], remote:str, model:str, reasoning_effort:str='high', connection_parallelism:int=64) -> list[dict]:

    # build the prompts
    
    prompts = [build_og_msg(d['context'], d['orig_question'], d['orig_answer']) for d in dataset]

    model = SglModelAsync(remote=remote, model=model, reasoning_effort=reasoning_effort, connection_parallelism=connection_parallelism)
    results, total_time = model.generate(prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    failed_responses = list()
    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            failed_responses.append(i)
            dataset[i]['reformat_question'] = None
            dataset[i]['reformat_answer'] = None
        else:
            parsed = answer_parser.parse_generated_open(res['content'])
            if parsed is None:
                dataset[i]['reformat_question'] = None
                dataset[i]['reformat_answer'] = None
                failed_responses.append(i)
            else:
                dataset[i]['reformat_question'] = parsed['question']
                dataset[i]['reformat_answer'] = parsed['correct_answer']

    if len(failed_responses) > 0:
        # raise Exception(f"Failed to rewrite {len(failed_responses)} questions")
        print(f"Failed to rewrite {len(failed_responses)} questions, removing them from the dataset")
        for i in sorted(failed_responses, reverse=True):
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
    parser.add_argument('--connection_parallelism', type=int, default=16, help='connection parallelism, set low for gpt-oss to try and avoid Harmony errors')

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
        if 'orig_question' not in item or 'orig_answer' not in item or 'context' not in item:
            raise ValueError('each element in the dataset must have the following keys: question, answer, context')

    print("Dataset has %d contexts" % len(dataset))

    dataset = reformat_questions(dataset, args.remote, args.model, args.reasoning_effort, args.connection_parallelism)

    elapsed_time = time.time() - start_time

    
    print(f"Saving {len(dataset)} questions to {output_fn}")
    with open(output_fn, 'w') as f:
        json.dump(dataset, f, indent=2)



