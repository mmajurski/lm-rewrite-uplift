"""Identify and subset questions that models find challenging, for targeted analysis."""

import os
import copy
import numpy as np
from inspect_ai import eval
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageUser,
)
from inspect_ai.scorer import Score, INCORRECT, Target
from inspect_ai.solver import Generate, Solver, solver, TaskState
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_FACT_TEMPLATE
# import vllm_inspect_provider
import utils
import time
import json
import random
import re

import model_interface

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


DEFAULT_GRADE_PATTERN = r"(?i)GRADE\s*:\s*([CPI])(.*)$"
"""Regex to extract the grade from the COT above."""




def extract_grade(response: str):
    # extract the grade
        match = re.search(DEFAULT_GRADE_PATTERN, response)
        if match:
            return match.group(1)
        else:
            return INCORRECT







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', type=str, default=os.getenv('VLLM_BASE_URL'), help='vLLM server URL, e.g. https://your-server:8443/v1')
    parser.add_argument('--model', type=str, default='google/gemma-3-27b-it')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()

    if args.example:
        main_dir = './data-example/'
    else:
        main_dir = './data-subset-100/'

    if args.model == 'G27B':
        model_name="google/gemma-3-27b-it"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-G27B/input/')
        main_ofp = os.path.join(main_dir, 'graded-G27B/')
    elif args.model == 'Q235B':
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-Q235B/input/')
        main_ofp = os.path.join(main_dir, 'graded-Q235B/')
    elif args.model == 'G4B':
        model_name="google/gemma-3-4b-it"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-G4B/input/')
        main_ofp = os.path.join(main_dir, 'graded-G4B/')
    elif args.model == 'L3B':
        model_name="meta-llama/Llama-3.2-3B-Instruct"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-L3B/input/')
        main_ofp = os.path.join(main_dir, 'graded-L3B/')
    elif args.model == 'Q7B':
        model_name="Qwen/Qwen2.5-7B-Instruct"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-Q7B/input/')
        main_ofp = os.path.join(main_dir, 'graded-Q7B/')
    elif args.model == 'L8B':
        model_name="meta-llama/Llama-3.1-8B-Instruct"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-L8B/input/')
        main_ofp = os.path.join(main_dir, 'graded-L8B/')
    elif args.model == 'G12B':
        model_name="google/gemma-3-12b-it"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-G12B/input/')
        main_ofp = os.path.join(main_dir, 'graded-G12B/')
    elif args.model == 'L70B':
        model_name="meta-llama/Llama-3.3-70B-Instruct"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-L70B/input/')
        main_ofp = os.path.join(main_dir, 'graded-L70B/')
    elif args.model == 'Phi4':
        model_name="microsoft/phi-4"
        remote = args.remote
        main_ifp = os.path.join(main_dir, 'graded-Phi4/input/')
        main_ofp = os.path.join(main_dir, 'graded-Phi4/')
    else:
        raise Exception(f"Model {args.model} not supported")



    model = model_interface.SglModelAsync(remote=remote, model=model_name)
    
    

    fldrs = [fn for fn in os.listdir(main_ifp) if os.path.isdir(os.path.join(main_ifp, fn))]
    for fldr in fldrs:
        ifp = os.path.join(main_ifp, fldr)
        ofp = os.path.join(main_ofp, fldr)


        os.makedirs(ofp, exist_ok=True)
        fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json')]

        for fn in fns:
            cur_ofp = os.path.join(ofp, fn)
            if os.path.exists(cur_ofp):
                print(f"Skipping {ifp}/{fn} because it already exists")
                continue

            print(f"Processing {ifp}/{fn}")
            with open(os.path.join(ifp, fn), 'r') as f:
                data = json.load(f)

            samples = data['samples']
            regrade_prompts = list()
            sample_idx = list()
            for i, sample in enumerate(samples):
                del sample['events']
                del sample['attachments']
                try:
                    grading = sample['scores']['model_graded_qa']
                    metadata = grading['metadata']
                    metadata_grading = metadata['grading']
                    grade_prompt = metadata_grading[0]['content']
                    regrade_prompts.append(grade_prompt)
                    sample_idx.append(i)
                except:
                    pass
            
            results, total_time = model.generate(regrade_prompts)
            for i in range(len(results)):
                results[i] = results[i]['content']
                if results[i] is None:
                    results[i] = 'Empty or Invalid model response'
            
            new_grades = list()
            for i in range(len(results)):
                res = results[i]
                k = sample_idx[i]
                g = extract_grade(res)
                new_grades.append(g == 'C')
                samples[k]['scores']['model_graded_qa']['metadata']['grading'][1]['content'] = results[i]
                samples[k]['scores']['model_graded_qa']['metadata']['grading'][1]['model'] = model_name
                samples[k]['scores']['model_graded_qa']['value'] = g

            acc = np.mean(new_grades)
            std_error = np.std(new_grades) / np.sqrt(len(new_grades))
            data['results']['scores'][0]['metrics']['accuracy']['value'] = float(acc)
            data['results']['scores'][0]['metrics']['stderr']['value'] = float(std_error)

            with open(cur_ofp,'w') as f:
                json.dump(data, f, indent=2)


