
import os
import copy
import numpy as np

import json
import re

import model_interface

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


plot_colors = [
    '#1f77b4',  # blue
    # '#ff7f0e',  # orange
    '#ffa655',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    # '#ffbb78',  # light orange
    '#ff7f0e',  # orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5'   # light cyan
]

plot_markers = [
    'o',
    '*',
    's',
    'D',
    'P',
    'X',
    'd',
    'H',
    'v',
    '^',
    '<',
    '>',
    '|',
    '_',
    '+',
    '.',
]



generating_model_name = 'Q235B'
# generating_model_name = 'gpt120b'
main_dir = './data-subset-500/'
ref_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}-filtered-orig')
reformat_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}-filtered-reformat')


ref_logs = [fn for fn in os.listdir(ref_question_folder) if fn.endswith('.json')]
reformat_logs = [fn for fn in os.listdir(reformat_question_folder) if fn.endswith('.json')]


if os.path.exists(f'impact_of_reformat_{generating_model_name}.json'):
    with open(f'impact_of_reformat_{generating_model_name}.json','r') as f:
        impact_of_reformat = json.load(f)
else:
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

        d_fp = data['eval']['task_args']['dataset_fldr'].replace('mmajursk','mmajurski')
        with open(d_fp, 'r') as f:
            source_dataset = json.load(f)

        if dataset_name not in impact_of_reformat:
            impact_of_reformat[dataset_name] = dict()
        
        samples = data['samples']
        for i, sample in enumerate(samples):
            question = sample['input']
            q_id = sample['id']-1  # counting starts from 1
            orig_question = source_dataset[q_id]['orig_question']
            if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
                acc = sample['scores']['model_graded_qa']['value'] == 'C'
                if orig_question not in impact_of_reformat[dataset_name]:
                    impact_of_reformat[dataset_name][orig_question] = dict()
                if model_name not in impact_of_reformat[dataset_name][orig_question]:
                    impact_of_reformat[dataset_name][orig_question][model_name] = dict()
                impact_of_reformat[dataset_name][orig_question][model_name]['ref_acc'] = acc    
            

    for fn_idx, fn in enumerate(reformat_logs):
        print(f"Processing {fn} ({fn_idx+1}/{len(reformat_logs)})")
        with open(os.path.join(reformat_question_folder, fn), 'r') as f:
            data = json.load(f)
        
        model_name = data['eval']['model'].replace("v_llm/", "")
        dataset_name = data['eval']['task_registry_name']

        d_fp = data['eval']['task_args']['dataset_fldr'].replace('mmajursk','mmajurski')
        with open(d_fp, 'r') as f:
            source_dataset = json.load(f)

        if dataset_name not in impact_of_reformat:
            impact_of_reformat[dataset_name] = dict()
        
        samples = data['samples']
        for i, sample in enumerate(samples):
            question = sample['input']
            q_id = sample['id']-1  # counting starts from 1
            orig_question = source_dataset[q_id]['orig_question']
            if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
                acc = sample['scores']['model_graded_qa']['value'] == 'C'
                if orig_question not in impact_of_reformat[dataset_name]:
                    impact_of_reformat[dataset_name][orig_question] = dict()
                if model_name not in impact_of_reformat[dataset_name][orig_question]:
                    impact_of_reformat[dataset_name][orig_question][model_name] = dict()
                impact_of_reformat[dataset_name][orig_question][model_name]['reformat_acc'] = acc
                # impact_of_reformat[dataset_name][orig_question][model_name]['new_question'] = question


    # Filter out any elements that don't have both reformat_acc and ref_acc
    filtered_impact = dict()
    for dataset_name, questions in impact_of_reformat.items():
        filtered_impact[dataset_name] = dict()
        for question, models in questions.items():
            filtered_impact[dataset_name][question] = dict()
            for model_name, scores in models.items():
                if 'ref_acc' in scores and 'reformat_acc' in scores:
                    scores['acc_uplift'] = float(scores['reformat_acc']) - float(scores['ref_acc'])
                    filtered_impact[dataset_name][question][model_name] = scores
                    

    # Replace the original dict with the filtered one
    impact_of_reformat = filtered_impact
    with open(f'impact_of_reformat_{generating_model_name}.json','w') as f:
        json.dump(impact_of_reformat, f, indent=2)


# Compute average acc_uplift per model and dataset
avg_uplift_per_model_dataset = dict()

for dataset_name, questions in impact_of_reformat.items():
    if dataset_name not in avg_uplift_per_model_dataset:
        avg_uplift_per_model_dataset[dataset_name] = dict()
    
    # Group questions by model to compute averages
    model_scores = dict()
    for question, models in questions.items():
        for model_name, scores in models.items():
            if model_name not in model_scores:
                model_scores[model_name] = {'uplifts': [], 'ref_accs': [], 'reformat_accs': []}
            model_scores[model_name]['uplifts'].append(scores['acc_uplift'])
            model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
            model_scores[model_name]['reformat_accs'].append(scores['reformat_acc'])
    
    # Compute averages for each model
    for model_name, score_lists in model_scores.items():
        avg_uplift = sum(score_lists['uplifts']) / len(score_lists['uplifts'])
        avg_ref_acc = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
        avg_reformat_acc = sum(score_lists['reformat_accs']) / len(score_lists['reformat_accs'])
        
        avg_uplift_per_model_dataset[dataset_name][model_name] = {
            'avg_acc_uplift': avg_uplift,
            'avg_ref_acc': avg_ref_acc,
            'avg_reformat_acc': avg_reformat_acc,
            'num_questions': len(score_lists['uplifts'])
        }

# Save the average uplift data
with open(f'avg_uplift_per_model_dataset_{generating_model_name}.json', 'w') as f:
    json.dump(avg_uplift_per_model_dataset, f, indent=2)


# Create scatterplots per model
import matplotlib.pyplot as plt
import numpy as np



# Get unique models across all datasets
all_models = set()
for dataset_name, models in avg_uplift_per_model_dataset.items():
    all_models.update(models.keys())

all_models = list(all_models)
all_models.sort()

# Create a scatterplot for each model
for model_name in all_models:
    plt.figure(figsize=(8, 8))

    ds_keys = list(avg_uplift_per_model_dataset.keys())
    ds_keys.sort()
    
    # Plot data points for each dataset
    for dataset_idx, dataset_name in enumerate(ds_keys):
        models = avg_uplift_per_model_dataset[dataset_name]

        if model_name in models:
            ref_acc = models[model_name]['avg_ref_acc']
            reformat_acc = models[model_name]['avg_reformat_acc']
            num_questions = models[model_name]['num_questions']
            
            marker = plot_markers[dataset_idx % len(plot_markers)]
            color = plot_colors[dataset_idx]
            plt.scatter(ref_acc, reformat_acc, 
                       marker=marker, color=color, s=100, alpha=0.8, 
                       label=f'{dataset_name} (n={num_questions})')
    
    # Add diagonal line (y=x) representing no change
    min_val = 0
    max_val = 1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No change')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Average Dataset Reference Accuracy')
    plt.ylabel('Average Dataset Reformat Accuracy')
    plt.title(f'Average Reference vs Reformat Accuracy: {model_name}')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    
    
    # Make the plot square to maintain aspect ratio
    plt.axis('equal')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the plot
    os.makedirs(f'./imgs/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/{generating_model_name}/scatterplot_{model_name.replace("/", "_")}.svg', dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nScatterplots saved for {len(all_models)} models")




