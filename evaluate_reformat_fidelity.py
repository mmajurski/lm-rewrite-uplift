import os
import json
import numpy as np
import sys




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
    # model_prompts = [prompts.REFORMAT_VALIDATION_PROMPT.format(context=d.get('context', ''), question1=d['orig_question'], answer1=d['orig_answer'], question2=d['question'], answer2=d['answer']) for d in dataset]
    model_prompts = [prompts.REFORMAT_VALIDATION_PROMPT.format(context=d.get('context', ''), question1=d['orig_question'], answer1=d['orig_answer'], question2=d['reformat_question'], answer2=d['reformat_answer']) for d in dataset]
    

    model = SglModelAsync(remote=remote, model=model, reasoning_effort='high')
    results, total_time = model.generate(model_prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    # reload the dataset to grab any potential changes to the dataset
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

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


    force_flag = False
    print(f"validate_reformat_fidelity: Validating reformat fidelity for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")

        compute_scores(dataset_fp, remote, model, force=force_flag)
    



    avg_question_similarity_score = {}
    avg_answer_similarity_score = {}
    for fn in fns:
        dataset_fp = f"{ifp}/{fn}"
        with open(dataset_fp, 'r') as f:
            data = json.load(f)
        
        
        avg_question_similarity_score[fn] = np.mean([d['reformat_question_similarity_score'] for d in data])
        avg_answer_similarity_score[fn] = np.mean([d['reformat_answer_similarity_score'] for d in data])
        failed_entries = [d for d in data if d.get('reformat_answer_similarity_score') < 5 or d.get('reformat_question_similarity_score') < 5]

        # if len(failed_entries) > 0:
        #     print(f"Dataset: {os.path.basename(fn)}")
        #     print(f"  Average reformat question similarity score: {avg_question_similarity_score[fn]:.4f}")
        #     print(f"  Average reformat answer similarity score: {avg_answer_similarity_score[fn]:.4f}")
        #     print(f"  Number of answer preservation failures: {len(failed_entries)}/{len(data)}")
        #     for d in failed_entries:
        #         print(f"    Orig Question: {d['orig_question']}")
        #         print(f"    Reformat Question: {d['reformat_question']}")
        #         print(f"    Reformat Question Similarity: {d.get('reformat_question_similarity_score', 'missing')}")
        #         print(f"    Orig Answer: {d.get('orig_answer', 'missing')}")
        #         print(f"    Reformat Answer: {d.get('reformat_answer', 'missing')}")
        #         print(f"    Reformat Answer Similarity: {d.get('reformat_answer_similarity_score', 'missing')}")
        #         print()
        #     print()





if __name__ == '__main__':
    remote="pn131285:8447"
    model = 'gpt-oss-120b'
    # remote = 'pn131285:8446'
    # model = 'Qwen/Qwen3-235B-A22B-Instruct-2507-FP8'


    # for model_name in ['gpt120b','Q235B']:
    #     ifp = f'./data-subset-500/oe-{model_name}/'
    #     print(f"Evaluating answer isomorphy features for {model} on the reformatted questions")
    #     validate_reformat_fidelity(ifp, remote, model)


    # for model_name in ['gpt120b','Q235B']:
    #     ifp = f'./data-post-cutoff/oe-{model_name}/'
    #     print(f"Evaluating answer isomorphy features for {model} on the reformatted questions")
    #     validate_reformat_fidelity(ifp, remote, model)

    for model_name in ['gpt20b', 'gpt120b', 'Q235B']:
        ifp = f'./data-subset-500-afc/oe-{model_name}-afc/'
        print(f"Evaluating answer isomorphy features for {model} on the reformatted questions")
        validate_reformat_fidelity(ifp, remote, model)

        ifp = f'./data-post-cutoff-afc/oe-{model_name}-afc/'
        print(f"Evaluating answer isomorphy features for {model} on the reformatted questions")
        validate_reformat_fidelity(ifp, remote, model)





