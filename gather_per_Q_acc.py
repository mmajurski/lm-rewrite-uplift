
import os
import copy
import numpy as np

import json
import re

import model_interface

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()





main_dir = './data-subset-1000/'
ref_question_folder = os.path.join(main_dir, 'logs-oe-unmodified')
reformat_question_folder = os.path.join(main_dir, 'logs-oe-Q235B-reformat')


ref_logs = [fn for fn in os.listdir(ref_question_folder) if fn.endswith('.json')]
reformat_logs = [fn for fn in os.listdir(reformat_question_folder) if fn.endswith('.json')]


impact_of_reformat = dict()
# top level key is the dataset name
# second level key is the question text
# third level key is the evaluation model name
# final key is ['ref_acc', 'reformat_acc'] with the associated accuracy values

for fn_idx, fn in enumerate(ref_logs):
    print(f"Processing {fn} ({fn_idx+1}/{len(ref_logs)})")
    with open(os.path.join(ref_question_folder, fn), 'r') as f:
        data = json.load(f)
    
    model_name = data['eval']['model'].replace("v_llm/", "")
    dataset_name = data['eval']['task_registry_name']
    if dataset_name not in impact_of_reformat:
        impact_of_reformat[dataset_name] = dict()
    
    samples = data['samples']
    for i, sample in enumerate(samples):
        question = sample['input']
        if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
            acc = sample['scores']['model_graded_qa']['value'] == 'C'
            if question not in impact_of_reformat[dataset_name]:
                impact_of_reformat[dataset_name][question] = dict()
            if model_name not in impact_of_reformat[dataset_name][question]:
                impact_of_reformat[dataset_name][question][model_name] = dict()
            impact_of_reformat[dataset_name][question][model_name]['ref_acc'] = acc
        

for fn_idx, fn in enumerate(reformat_logs):
    print(f"Processing {fn} ({fn_idx+1}/{len(reformat_logs)})")
    with open(os.path.join(reformat_question_folder, fn), 'r') as f:
        data = json.load(f)
    
    model_name = data['eval']['model'].replace("v_llm/", "")
    dataset_name = data['eval']['task_registry_name']
    if dataset_name not in impact_of_reformat:
        impact_of_reformat[dataset_name] = dict()
    
    samples = data['samples']
    for i, sample in enumerate(samples):
        question = sample['input']
        if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
            acc = sample['scores']['model_graded_qa']['value'] == 'C'
            if question not in impact_of_reformat[dataset_name]:
                impact_of_reformat[dataset_name][question] = dict()
            if model_name not in impact_of_reformat[dataset_name][question]:
                impact_of_reformat[dataset_name][question][model_name] = dict()
            impact_of_reformat[dataset_name][question][model_name]['reformat_acc'] = acc


with open('impact_of_reformat.json','w') as f:
    json.dump(impact_of_reformat, f, indent=2)


