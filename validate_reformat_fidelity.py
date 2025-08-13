import os
import json
import numpy as np
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import prompts
import answer_parser
from model_interface import SglModelAsync




def compute_scores(dataset_fp, remote, model, force=False):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

    any_missing = False
    for d in dataset:
        if 'reformat_question_similarity_score' not in d or d['reformat_question_similarity_score'] is None:
            any_missing = True
        if 'reformat_answer_similarity_score' not in d or d['reformat_answer_similarity_score'] is None:
            any_missing = True
    if not any_missing and not force:
        return
    
    for d in dataset:
        if 'orig_question' not in d or 'orig_answer' not in d:
            return
    
    print(f"Computing meta properties for {dataset_fp}")
    print("Dataset has %d contexts" % len(dataset))

    # build the prompts
    model_prompts = [prompts.REFORMAT_VALIDATION_PROMPT.format(context=d.get('context', ''), question1=d['orig_question'], answer1=d['orig_answer'], question2=d['question'], answer2=d['answer']) for d in dataset]
    

    model = SglModelAsync(remote=remote, model=model, reasoning_effort='high')
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
            parsed = answer_parser.parse_reformat_validity_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
                dataset[i]['reformat_question_similarity_score'] = None  
                dataset[i]['reformat_answer_similarity_score'] = None
            else:
                dataset[i]['reformat_question_similarity_score'] = parsed['question_similarity_score']
                dataset[i]['reformat_answer_similarity_score'] = parsed['answer_similarity_score']


    scores = [d['reformat_question_similarity_score'] for d in dataset]
    print(f"Average reformat question similarity score: {np.mean(scores)}")

    scores = [d['reformat_answer_similarity_score'] for d in dataset]
    print(f"Average reformat answer similarity score: {np.mean(scores)}")

    print(f"Saving {len(dataset)} questions to {dataset_fp}")
    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)
    return








def validate_reformat_fidelity(ifp, remote:str, model:str):

    fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json') and 'failed' not in fn]
    fns.sort()
    if len(fns) == 0:
        print(f"validate_reformat_fidelity: No datasets found in {ifp}")
        return
    print(f"validate_reformat_fidelity: Found {len(fns)} datasets")
    print(f"validate_reformat_fidelity: Datasets: {fns}")


    force_flag = True
    print(f"validate_reformat_fidelity: Validating reformat fidelity for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")

        compute_scores(dataset_fp, remote, model, force=force_flag)
    



    avg_question_similarity_score = {}
    avg_answer_similarity_score = {}
    for fn in fns:
        if 'sec_qa_reformat' in fn:
            continue
        dataset_fp = f"{ifp}/{fn}"
        with open(dataset_fp, 'r') as f:
            data = json.load(f)

        invalid_idx = list()
        for i in range(len(data)):
            d = data[i]
            if d.get('reformat_question_similarity_score') is None or not (1 <= d['reformat_question_similarity_score'] <= 10):
                invalid_idx.append(i)
            if d.get('reformat_answer_similarity_score') is None or not (1 <= d['reformat_answer_similarity_score'] <= 10):
                invalid_idx.append(i)

        invalid_idx = list(set(invalid_idx))
        
        # Print invalid data entries
        if invalid_idx:
            print(f"Found {len(invalid_idx)} invalid entries in dataset {fn}:")
            for idx in invalid_idx: 
                print(f"  Entry {idx}:")
                print(f"    Question: {data[idx]['question']}")
                print(f"    Reformat Question Similarity: {data[idx].get('reformat_question_similarity_score', 'missing')}")
                print(f"    Reformat Answer Similarity: {data[idx].get('reformat_answer_similarity_score', 'missing')}")
                print()

            raise Exception(f"Found {len(invalid_idx)} invalid entries in dataset {fn}")
        
        
        avg_question_similarity_score[fn] = np.mean([d['reformat_question_similarity_score'] for d in data])
        avg_answer_similarity_score[fn] = np.mean([d['reformat_answer_similarity_score'] for d in data])
        nb_failures = sum([1 for d in data if d.get('reformat_answer_similarity_score') < 5])
        failed_entries = [d for d in data if d.get('reformat_answer_similarity_score') < 5]

        print(f"Dataset: {os.path.basename(fn)}")
        print(f"  Average reformat question similarity score: {avg_question_similarity_score[fn]:.4f}")
        print(f"  Average reformat answer similarity score: {avg_answer_similarity_score[fn]:.4f}")
        print(f"  Number of answer preservation failures: {nb_failures}")
        for d in failed_entries:
            print(f"    Question: {d['question']}")
            print(f"    Reformat Question Similarity: {d.get('reformat_question_similarity_score', 'missing')}")
            print(f"    Reformat Answer Similarity: {d.get('reformat_answer_similarity_score', 'missing')}")
            print(f"    GT Answer: {d.get('orig_answer', 'missing')}")
            print(f"    Reformat Answer: {d.get('answer', 'missing')}")
            print()
        print()



if __name__ == '__main__':
    ifp = './data-subset-1000/oe-gpt120b/'

    model="gpt-oss-120b"
    remote="pn131285:8447"

    # model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    # remote="pn131285:8446"
    
    validate_reformat_fidelity(ifp, remote, model)







