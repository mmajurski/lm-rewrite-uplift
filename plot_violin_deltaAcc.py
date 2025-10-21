
import os
import copy
import numpy as np

import json
from scipy.stats import gaussian_kde


# plot_colors = [
#     '#1f77b4',  # blue
#     '#ffa655',  # orange
#     '#2ca02c',  # green
#     '#d62728',  # red
#     '#9467bd',  # purple
#     '#8c564b',  # brown
#     '#e377c2',  # pink
#     '#7f7f7f',  # gray
#     '#bcbd22',  # olive
#     '#17becf',  # cyan
#     '#aec7e8',  # light blue
#     '#ff7f0e',  # orange
#     '#98df8a',  # light green
#     '#ff9896',  # light red
#     '#c5b0d5',  # light purple
#     '#c49c94',  # light brown
#     '#f7b6d2',  # light pink
#     '#c7c7c7',  # light gray
#     '#dbdb8d',  # light olive
#     '#9edae5',  # light cyan
#     # Additional colors
#     '#393b79',  # dark blue
#     '#637939',  # dark green
#     '#8c6d31',  # dark brown
#     '#843c39',  # dark red
#     '#7b4173',  # dark purple
#     '#17becf',  # cyan (repeat for more variety)
#     '#bc80bd',  # lavender
#     '#ffed6f',  # yellow
#     '#1b9e77',  # teal
#     '#e7298a',  # magenta
#     '#66a61e',  # olive green
#     '#e6ab02',  # mustard
#     '#a6761d',  # brown
#     '#666666',  # dark gray
# ]


# Create a list of 3 maximally visually separable colors for plotting
plot_colors = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
]


for generating_model_name in ['gpt120b', 'gpt20b', 'Q235B']:
    impact_of_giveaway = dict()
    for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc','data-subset-500-afc']:
    # for dataset_fldr in ['data-post-cutoff', 'data-post-cutoff-afc']:
        print(f"Processing {dataset_fldr}")
        
            
        main_dir = f'./{dataset_fldr}/'

        if dataset_fldr.endswith('-afc'):
            context_type = 'afc'
            context_str = '-afc'
        else:
            context_type = 'orig'
            context_str = ""
        
        q_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig')
        qC_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig-giveaway')
        r_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat') 
        r_giveaway_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat-giveaway')

        if os.path.exists(q_folder):
            q_logs = [fn for fn in os.listdir(q_folder) if fn.endswith('.json')]
        else:
            q_logs = []
        if os.path.exists(qC_folder):
            qC_logs = [fn for fn in os.listdir(qC_folder) if fn.endswith('.json')]
        else:
            qC_logs = []
        if os.path.exists(r_folder):
            r_logs = [fn for fn in os.listdir(r_folder) if fn.endswith('.json')]
        else:
            r_logs = []
        if os.path.exists(r_giveaway_folder):
            r_giveaway_logs = [fn for fn in os.listdir(r_giveaway_folder) if fn.endswith('.json')]
        else:
            r_giveaway_logs = []


        for log_type in ['ref','ref_giveaway', 'rewrite', 'rewrite_giveaway']:
            if log_type == 'ref':
                logs = q_logs
                question_folder = q_folder
            elif log_type == 'ref_giveaway':
                logs = qC_logs 
                question_folder = qC_folder
            elif log_type == 'rewrite':
                logs = r_logs
                question_folder = r_folder
            elif log_type == 'rewrite_giveaway':
                logs = r_giveaway_logs
                question_folder = r_giveaway_folder
            else:
                raise ValueError(f"Invalid log type: {log_type}")


            for fn_idx, fn in enumerate(logs):
                # print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
                with open(os.path.join(question_folder, fn), 'r') as f:
                    data = json.load(f)
                
                model_name = data['eval']['model'].replace("v_llm/", "")
                model_name = model_name.split('/')[1] if '/' in model_name else model_name
                if 'gpt-5' in model_name:
                    continue  # remove the gpt5 data as its mostly just HLE
                dataset_name = data['eval']['task_registry_name']

                d_fp = data['eval']['task_args']['dataset_fldr']
                # d_fp = d_fp.replace('mmajursk','mmajurski')
                d_fp = d_fp.replace('/home/mmajursk/github/lm-rewrite-uplift','/Users/mmajursk/github/lm-rewrite-uplift')
                with open(d_fp, 'r') as f:
                    source_dataset = json.load(f)

                if dataset_name not in impact_of_giveaway:
                    impact_of_giveaway[dataset_name] = dict()
                
                samples = data['samples']
                for q_id, sample in enumerate(samples):
                    # question = sample['input']
                    orig_question = source_dataset[q_id]['orig_question']
                    reformat_question = source_dataset[q_id]['reformat_question']
                    reformat_answer = source_dataset[q_id]['reformat_answer']
                    orig_answer = source_dataset[q_id]['orig_answer']
                    context = source_dataset[q_id]['context']
                    if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
                        acc = sample['scores']['model_graded_qa']['value'] == 'C'
                        if orig_question not in impact_of_giveaway[dataset_name]:
                            impact_of_giveaway[dataset_name][orig_question] = dict()
                            impact_of_giveaway[dataset_name][orig_question]['orig_question'] = orig_question
                            impact_of_giveaway[dataset_name][orig_question]['orig_answer'] = orig_answer
                            impact_of_giveaway[dataset_name][orig_question]['context'] = context
                            impact_of_giveaway[dataset_name][orig_question]['models'] = dict()
                        if model_name not in impact_of_giveaway[dataset_name][orig_question]['models']:
                            impact_of_giveaway[dataset_name][orig_question]['models'][model_name] = dict()
                        if context_type == 'orig':
                            if log_type == 'ref':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc'] = acc 
                            elif log_type == 'ref_giveaway':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc_giveaway'] = acc 
                            elif log_type == 'rewrite':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['rewrite_acc'] = acc 
                            elif log_type == 'rewrite_giveaway':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['rewrite_acc_giveaway'] = acc 
                            else:
                                raise ValueError(f"Invalid log type: {log_type}")
                        else:
                            if log_type == 'ref':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc_afc'] = acc
                            elif log_type == 'ref_giveaway':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc_giveaway_afc'] = acc
                            elif log_type == 'rewrite':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['rewrite_acc_afc'] = acc
                            elif log_type == 'rewrite_giveaway':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['rewrite_acc_giveaway_afc'] = acc
                            else:
                                raise ValueError(f"Invalid log type: {log_type}")
                                    




    # Compute average acc_uplift per model and dataset
    avg_uplift_per_model_dataset = dict()

    for dataset_name, questions in impact_of_giveaway.items():
        if dataset_name not in avg_uplift_per_model_dataset:
            avg_uplift_per_model_dataset[dataset_name] = dict()
        
        # Group questions by model to compute averages
        model_scores = dict()
        for question, sub_data in questions.items():
            models = sub_data['models']
            for model_name, scores in models.items():
                if model_name not in model_scores:
                    model_scores[model_name] = {'ref_accs': [], 'ref_accs_giveaway': [], 'rewrite_accs': [], 'rewrite_accs_giveaway': [], 'ref_accs_afc': [], 'ref_accs_giveaway_afc': [], 'rewrite_accs_afc': [], 'rewrite_accs_giveaway_afc': []}
                if 'ref_acc' in scores and 'ref_acc_giveaway' in scores and 'rewrite_acc' in scores and 'rewrite_acc_giveaway' in scores:
                    model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                    model_scores[model_name]['ref_accs_giveaway'].append(scores['ref_acc_giveaway'])
                    model_scores[model_name]['rewrite_accs'].append(scores['rewrite_acc'])
                    model_scores[model_name]['rewrite_accs_giveaway'].append(scores['rewrite_acc_giveaway'])
                if 'ref_acc_afc' in scores and 'ref_acc_giveaway_afc' in scores and 'rewrite_acc_afc' in scores and 'rewrite_acc_giveaway_afc' in scores:
                    model_scores[model_name]['ref_accs_afc'].append(scores['ref_acc_afc'])
                    model_scores[model_name]['ref_accs_giveaway_afc'].append(scores['ref_acc_giveaway_afc'])
                    model_scores[model_name]['rewrite_accs_afc'].append(scores['rewrite_acc_afc'])
                    model_scores[model_name]['rewrite_accs_giveaway_afc'].append(scores['rewrite_acc_giveaway_afc'])
        
        # Compute averages for each model
        for model_name, score_lists in model_scores.items():
            if len(score_lists['ref_accs']) > 0:
                avg_ref_acc = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
                avg_ref_acc_giveaway = sum(score_lists['ref_accs_giveaway']) / len(score_lists['ref_accs_giveaway'])
                avg_rewrite_acc = sum(score_lists['rewrite_accs']) / len(score_lists['rewrite_accs'])
                avg_rewrite_acc_giveaway = sum(score_lists['rewrite_accs_giveaway']) / len(score_lists['rewrite_accs_giveaway'])
                avg_ref_acc_afc = sum(score_lists['ref_accs_afc']) / len(score_lists['ref_accs_afc'])
                avg_ref_acc_giveaway_afc = sum(score_lists['ref_accs_giveaway_afc']) / len(score_lists['ref_accs_giveaway_afc'])
                avg_rewrite_acc_afc = sum(score_lists['rewrite_accs_afc']) / len(score_lists['rewrite_accs_afc'])
                avg_rewrite_acc_giveaway_afc = sum(score_lists['rewrite_accs_giveaway_afc']) / len(score_lists['rewrite_accs_giveaway_afc'])
                avg_uplift_per_model_dataset[dataset_name][model_name] = {
                    'avg_ref_acc': avg_ref_acc,
                    'avg_ref_acc_giveaway': avg_ref_acc_giveaway,
                    'avg_rewrite_acc': avg_rewrite_acc,
                    'avg_rewrite_acc_giveaway': avg_rewrite_acc_giveaway,
                    'avg_ref_acc_afc': avg_ref_acc_afc,
                    'avg_ref_acc_giveaway_afc': avg_ref_acc_giveaway_afc,
                    'avg_rewrite_acc_afc': avg_rewrite_acc_afc,
                    'avg_rewrite_acc_giveaway_afc': avg_rewrite_acc_giveaway_afc
                }



    # Create scatterplots per model
    import matplotlib.pyplot as plt
    import numpy as np



    # Get unique models across all datasets
    all_models = set()
    all_datasets = set()
    for dataset_name, models in avg_uplift_per_model_dataset.items():
        all_models.update(models.keys())
        all_datasets.add(dataset_name)


    all_datasets = list(all_datasets)
    all_datasets.sort()
    all_models = list(all_models)
    all_models.sort()


    
    


    per_model_data = {}
    for m_idx, model_name in enumerate(all_models): 
        per_model_data[model_name] = {}

        # Collect accuracy values for each dataset
        for dataset_name in all_datasets:
            if dataset_name not in avg_uplift_per_model_dataset:
                continue
                
            models = avg_uplift_per_model_dataset[dataset_name]
            if model_name not in models:
                continue
                
            model_data = models[model_name]
            r_minus_q = model_data['avg_rewrite_acc'] - model_data['avg_ref_acc']
            r_minus_q_afc_giveaway = model_data['avg_rewrite_acc'] - model_data['avg_ref_acc_giveaway_afc']
            r_minus_q_giveaway = model_data['avg_rewrite_acc'] - model_data['avg_ref_acc_giveaway']
            if 'r_minus_q' not in per_model_data[model_name]:
                per_model_data[model_name]['r_minus_q'] = []
            if 'r_minus_q_afc_giveaway' not in per_model_data[model_name]:
                per_model_data[model_name]['r_minus_q_afc_giveaway'] = []
            if 'r_minus_q_giveaway' not in per_model_data[model_name]:
                per_model_data[model_name]['r_minus_q_giveaway'] = []
            per_model_data[model_name]['r_minus_q'].append(r_minus_q)
            per_model_data[model_name]['r_minus_q_afc_giveaway'].append(r_minus_q_afc_giveaway)
            per_model_data[model_name]['r_minus_q_giveaway'].append(r_minus_q_giveaway)


    # Violin plot: for each model, show the distribution of r_minus_q per dataset
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data: each violin uses per_model_data[model_name]['r_minus_q']
    violin_data = [per_model_data[model]['r_minus_q'] for model in all_models]

    # Make the violin plot
    parts = ax.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    # Set x-ticks and labels (models)
    ax.set_xticks(np.arange(1, len(all_models) + 1))
    ax.set_xticklabels(all_models, rotation=45, ha='right')

    ax.set_ylabel("Benchmark Accuracy:\nRewrite - Ref")
    ax.set_title("Distribution of Benchmark Accuracy Uplift per Model (All Datasets)")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    os.makedirs(f'./imgs/acc_uplift/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/acc_uplift/{generating_model_name}/r_minus_q.svg', dpi=300, bbox_inches='tight')
    plt.close()


    # Violin plot: for each model, show the distribution of r_minus_q per dataset
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data: each violin uses per_model_data[model_name]['r_minus_q']
    violin_data = [per_model_data[model]['r_minus_q_afc_giveaway'] for model in all_models]

    # Make the violin plot
    parts = ax.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    # Set x-ticks and labels (models)
    ax.set_xticks(np.arange(1, len(all_models) + 1))
    ax.set_xticklabels(all_models, rotation=45, ha='right')

    ax.set_ylabel("Benchmark Accuracy:\nRewrite - (Ref with AFC)")
    ax.set_title("Distribution of Benchmark Accuracy Uplift per Model (All Datasets)")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    os.makedirs(f'./imgs/acc_uplift/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/acc_uplift/{generating_model_name}/r_minus_q_afc_giveaway.svg', dpi=300, bbox_inches='tight')
    plt.close()




    # Violin plot: for each model, show the distribution of r_minus_q per dataset
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data: each violin uses per_model_data[model_name]['r_minus_q']
    violin_data = [per_model_data[model]['r_minus_q_giveaway'] for model in all_models]

    # Make the violin plot
    parts = ax.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    # Set x-ticks and labels (models)
    ax.set_xticks(np.arange(1, len(all_models) + 1))
    ax.set_xticklabels(all_models, rotation=45, ha='right')

    ax.set_ylabel("Benchmark Accuracy:\nRewrite - (Ref with Context)")
    ax.set_title("Distribution of Benchmark Accuracy Uplift per Model (All Datasets)")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    os.makedirs(f'./imgs/acc_uplift/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/acc_uplift/{generating_model_name}/r_minus_q_giveaway.svg', dpi=300, bbox_inches='tight')
    plt.close()


    




    per_dataset_data = {}
    for d_idx, dataset_name in enumerate(all_datasets):
        per_dataset_data[dataset_name] = {}

        # Collect accuracy values for each model
        for model_name in all_models:
            if dataset_name not in avg_uplift_per_model_dataset:
                continue

            models = avg_uplift_per_model_dataset[dataset_name]
            if model_name not in models:
                continue

            model_data = models[model_name]
            r_minus_q = model_data['avg_rewrite_acc'] - model_data['avg_ref_acc']
            r_minus_q_afc_giveaway = model_data['avg_rewrite_acc'] - model_data['avg_ref_acc_giveaway_afc']
            r_minus_q_giveaway = model_data['avg_rewrite_acc'] - model_data['avg_ref_acc_giveaway']
            if 'r_minus_q' not in per_dataset_data[dataset_name]:
                per_dataset_data[dataset_name]['r_minus_q'] = []
            if 'r_minus_q_afc_giveaway' not in per_dataset_data[dataset_name]:
                per_dataset_data[dataset_name]['r_minus_q_afc_giveaway'] = []
            if 'r_minus_q_giveaway' not in per_dataset_data[dataset_name]:
                per_dataset_data[dataset_name]['r_minus_q_giveaway'] = []
            per_dataset_data[dataset_name]['r_minus_q'].append(r_minus_q)
            per_dataset_data[dataset_name]['r_minus_q_afc_giveaway'].append(r_minus_q_afc_giveaway)
            per_dataset_data[dataset_name]['r_minus_q_giveaway'].append(r_minus_q_giveaway)

        if 'hle' in dataset_name:
            print(per_dataset_data[dataset_name])

    

    # Violin plot: for each model, show the distribution of r_minus_q per dataset
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data: each violin uses per_model_data[model_name]['r_minus_q']
    violin_data = [per_dataset_data[dataset_name]['r_minus_q'] for dataset_name in all_datasets]

    # Make the violin plot
    parts = ax.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    # Set x-ticks and labels (models)
    ax.set_xticks(np.arange(1, len(all_datasets) + 1))
    ax.set_xticklabels(all_datasets, rotation=45, ha='right')

    ax.set_ylabel("Benchmark Accuracy:\nRewrite - Ref")
    ax.set_title("Distribution of Benchmark Accuracy Uplift per Dataset (All Models)")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    os.makedirs(f'./imgs/acc_uplift/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/acc_uplift/{generating_model_name}/r_minus_q_dataset.svg', dpi=300, bbox_inches='tight')
    plt.close()



    # Violin plot: for each model, show the distribution of r_minus_q per dataset
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data: each violin uses per_model_data[model_name]['r_minus_q']
    violin_data = [per_dataset_data[dataset_name]['r_minus_q_afc_giveaway'] for dataset_name in all_datasets]

    # Make the violin plot
    parts = ax.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    # Set x-ticks and labels (models)
    ax.set_xticks(np.arange(1, len(all_datasets) + 1))
    ax.set_xticklabels(all_datasets, rotation=45, ha='right')

    ax.set_ylabel("Benchmark Accuracy:\nRewrite - (Ref with AFC)")
    ax.set_title("Distribution of Benchmark Accuracy Uplift per Dataset (All Models)")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    os.makedirs(f'./imgs/acc_uplift/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/acc_uplift/{generating_model_name}/r_minus_q_afc_giveaway_dataset.svg', dpi=300, bbox_inches='tight')
    plt.close()




    # Violin plot: for each model, show the distribution of r_minus_q per dataset
    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data: each violin uses per_model_data[model_name]['r_minus_q']
    violin_data = [per_dataset_data[dataset_name]['r_minus_q_giveaway'] for dataset_name in all_datasets]

    # Make the violin plot
    parts = ax.violinplot(
        violin_data,
        showmeans=True,
        showextrema=True
    )

    # Set x-ticks and labels (models)
    ax.set_xticks(np.arange(1, len(all_datasets) + 1))
    ax.set_xticklabels(all_datasets, rotation=45, ha='right')

    ax.set_ylabel("Benchmark Accuracy:\nRewrite - (Ref with Context)")
    ax.set_title("Distribution of Benchmark Accuracy Uplift per Dataset (All Models)")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)

    plt.tight_layout()
    os.makedirs(f'./imgs/acc_uplift/{generating_model_name}', exist_ok=True)
    plt.savefig(f'./imgs/acc_uplift/{generating_model_name}/r_minus_q_giveaway_dataset.svg', dpi=300, bbox_inches='tight')
    plt.close()