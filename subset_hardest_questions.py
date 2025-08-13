
import os
import copy
import numpy as np
from inspect_ai import eval
from inspect_ai.scorer import Score, INCORRECT, Target
# import vllm_inspect_provider

import json
import re

import model_interface

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



dataset_names = ['flashrag_2wikimultihopqa', 'flashrag_boolq', 'flashrag_fermi', 'flashrag_hotpotqa', 'flashrag_msmarcoqa', 'flashrag_musique', 'mrqa_HotpotQA', 'mrqa_NaturalQuestionsShort', 'mrqa_TriviaQA-web', 'pubmed_qa', 'squadv2', 'triva_qa']


main_dir = './data-subset-1000/'
fldr = 'oe-unmodified'

data_fp = os.path.join(main_dir, fldr)
log_fp = os.path.join(main_dir, f'logs-{fldr}/')
ofp = os.path.join(main_dir, f'{fldr}-subset/')
                        
os.makedirs(ofp, exist_ok=True)



for dataset_name in dataset_names:
    data_json_fn = f'{dataset_name}.json'
    data_json_fp = os.path.join(data_fp, data_json_fn)

    with open(data_json_fp, 'r') as f:
        question_data = json.load(f)

    

    question_accuracy = dict()
    question_correct_models = dict()
    question_incorrect_models = dict()

    fns = [fn for fn in os.listdir(log_fp) if fn.endswith('.json') and dataset_name.replace('_','-') in fn]

    for fn in fns:
            
        with open(os.path.join(log_fp, fn), 'r') as f:
            log_data = json.load(f)

        model_name = log_data['eval']['model'].replace('v_llm/','')
        # print(f"Processing {model_name} for {dataset_name}")
        # print(f'log file: {fn}')

        samples = log_data['samples']
        for i, sample in enumerate(samples):
            question = sample['input']
            try:
                grading = sample['scores']['model_graded_qa']['value'] == 'C'
                if question not in question_accuracy:
                    question_accuracy[question] = list()
                    question_correct_models[question] = list()
                    question_incorrect_models[question] = list()
                question_accuracy[question].append(grading)
                if grading:
                    question_correct_models[question].append(model_name)
                else:
                    question_incorrect_models[question].append(model_name)
            except:
                pass

    
    for sample in question_data:
        question = sample['question']
        grading = question_accuracy[question]
        accuracy = np.mean(grading)
        sample['ensemble_accuraacy'] = accuracy
        sample['correct_models'] = question_correct_models[question]
        sample['incorrect_models'] = question_incorrect_models[question]

    # Get the bottom 200 questions in accuracy
    # Sort question_data by 'ensemble_accuraacy' in ascending order
    question_data = sorted(question_data, key=lambda x: x['ensemble_accuraacy'])
    # Take the bottom 200
    question_data = question_data[:200]

    vals = [s['ensemble_accuraacy'] for s in question_data]
    mv = np.mean(vals)
    print(f'{dataset_name} mean accuracy: {mv}')

    cur_ofp = os.path.join(ofp, data_json_fn)
    with open(cur_ofp,'w') as f:
        json.dump(question_data, f, indent=2)


