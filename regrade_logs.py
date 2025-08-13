
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
    args = parser.parse_args()

    
    main_dir = './data-subset-1000/'

    
    # model_name="gpt-oss-120b"
    # remote="pn131285:8447"
    model_name="gpt-oss-20b"
    remote="pn131285:8443"

    ifp = os.path.join(main_dir, 'logs-oe-unmodified-regrade20-d/input/')
    ofp = os.path.join(main_dir, 'logs-oe-unmodified-regrade20-d/')
    os.makedirs(ofp, exist_ok=True)


    model = model_interface.SglModelAsync(remote=remote, model=model_name)
    
    
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
            if 'events' in sample:
                del sample['events']
            if 'attachments' in sample:
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


