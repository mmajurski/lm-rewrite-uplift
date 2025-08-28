
import os
import copy
import numpy as np

import json
import re






main_dir = './data-subset-500'
# main_dir = './data-post-cutoff'
fldrs = [fn for fn in os.listdir(main_dir) if fn.startswith('logs-')]
fldrs.sort()
for fldr in fldrs:
    json_fns = [fn for fn in os.listdir(os.path.join(main_dir, fldr)) if fn.endswith('.json')]
    json_fns.sort()
    for json_fn in json_fns:

        with open(os.path.join(main_dir, fldr, json_fn), 'r') as f:
            data = json.load(f)

        dataset_path = data['eval']['task_args']['dataset_fldr'].replace('mmajurski','mmajursk')
        with open(dataset_path, 'r') as f:
            source_dataset = json.load(f)
        if 'orig' in fldr:
            source_questions = [d['orig_question'] for d in source_dataset]
        elif 'reformat' in fldr:
            source_questions = [d['question'] for d in source_dataset]
        else:
            raise RuntimeError
        

        samples = data['samples']
        samples_to_delete = []
        for i, sample in enumerate(samples):
            valid = False
            for q in source_questions:
                if q in sample['input']:
                    valid = True
                    break
            if not valid:
                samples_to_delete.append(i)

        if len(samples_to_delete) > 0:
            # Delete all samples at once by creating a new list excluding the indices to delete
            data['samples'] = [s for idx, s in enumerate(data['samples']) if idx not in samples_to_delete]

            data['results']['total_samples'] = len(data['samples'])
            valid_scores = [1 if 'model_graded_qa' in s['scores'] else 0 for s in data['samples']]
            data['results']['completed_samples'] = sum(valid_scores)
            scores = [s['scores']['model_graded_qa']['value'] == 'C' if 'model_graded_qa' in s['scores'] else None for s in data['samples']]
            scores = np.asarray([s for s in scores if s is not None])
            avg_score = np.mean(scores)
            std_score = np.std(scores)/float(len(scores))
            data['results']['scores'][0]['metrics']['accuracy']['value'] = avg_score
            data['results']['scores'][0]['metrics']['stderr']['value'] = std_score
            data['eval']['dataset']['samples'] = len(scores)
            data['eval']['dataset']['sample_ids'] = list(range(len(scores)))

            out_fldr = main_dir.replace("500","500-cleaned").replace("cutoff","cutoff-cleaned")
            cur_ofp = os.path.join(out_fldr, fldr)
            os.makedirs(cur_ofp, exist_ok=True)
            with open(os.path.join(out_fldr, fldr, json_fn), 'w') as f:
                json.dump(data, f, indent=2)

