"""Distribution plots comparing original question vs question-with-context pairs."""

import os
import copy
import numpy as np

import json


# 



from plot_config import COLORS as plot_colors, MARKERS as plot_markers
impact_of_giveaway = dict()
for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc','data-subset-500-afc']:
    print(f"Processing {dataset_fldr}")
    for question_source in ['orig']:
        for generating_model_name in ['gpt120b']:
            
            main_dir = f'./{dataset_fldr}/'

            if dataset_fldr.endswith('-afc'):
                context_type = 'afc'
                context_str = '-afc'
            else:
                context_type = 'orig'
                context_str = ""
            
            base_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-{question_source}')
            giveaway_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-{question_source}-giveaway')

            if os.path.exists(base_question_folder):
                ref_logs = [fn for fn in os.listdir(base_question_folder) if fn.endswith('.json')]
            else:
                ref_logs = []
            if os.path.exists(giveaway_question_folder):
                giveaway_logs = [fn for fn in os.listdir(giveaway_question_folder) if fn.endswith('.json')]
            else:
                giveaway_logs = []


            for log_type in ['ref', 'giveaway']:
                if log_type == 'ref':
                    logs = ref_logs
                    question_folder = base_question_folder
                else:
                    logs = giveaway_logs
                    question_folder = giveaway_question_folder


                for fn_idx, fn in enumerate(logs):
                    # print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
                    with open(os.path.join(question_folder, fn), 'r') as f:
                        data = json.load(f)
                    
                    model_name = data['eval']['model'].replace("v_llm/", "")
                    model_name = model_name.split('/')[1] if '/' in model_name else model_name
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
                                else:
                                    impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['giveaway_acc'] = acc
                            else:
                                if log_type == 'ref':
                                    impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc_afc'] = acc
                                else:
                                    impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['giveaway_acc_afc'] = acc
                                




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
                model_scores[model_name] = {'ref_accs': [], 'giveaway_accs': [], 'ref_accs_afc': [], 'giveaway_accs_afc': []}
            if 'ref_acc' in scores and 'giveaway_acc' in scores:
                model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                model_scores[model_name]['giveaway_accs'].append(scores['giveaway_acc'])
            if 'ref_acc_afc' in scores and 'giveaway_acc_afc' in scores:
                model_scores[model_name]['ref_accs_afc'].append(scores['ref_acc_afc'])
                model_scores[model_name]['giveaway_accs_afc'].append(scores['giveaway_acc_afc'])
    
    # Compute averages for each model
    for model_name, score_lists in model_scores.items():
        if len(score_lists['ref_accs']) > 0:
            avg_ref_acc = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
            avg_giveaway_acc = sum(score_lists['giveaway_accs']) / len(score_lists['giveaway_accs'])
            avg_ref_acc_afc = sum(score_lists['ref_accs_afc']) / len(score_lists['ref_accs_afc'])
            avg_giveaway_acc_afc = sum(score_lists['giveaway_accs_afc']) / len(score_lists['giveaway_accs_afc'])
            avg_uplift_per_model_dataset[dataset_name][model_name] = {
                'avg_ref_acc': avg_ref_acc,
                'avg_giveaway_acc': avg_giveaway_acc,
                'avg_ref_acc_afc': avg_ref_acc_afc,
                'avg_giveaway_acc_afc': avg_giveaway_acc_afc,
                'num_questions': len(score_lists['ref_accs'])
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


# Create a scatterplot for each model
for m_idx, model_name in enumerate(all_models):            

    plt.figure(figsize=(4, 2.5))
    model_acc_vals_q = []
    model_acc_vals_qC = []
    model_acc_vals_qAfC = []

    
    # Plot data points for each dataset
    for d_idx, dataset_name in enumerate(all_datasets):
        if not dataset_name in avg_uplift_per_model_dataset.keys():
            continue
        models = avg_uplift_per_model_dataset[dataset_name]                    

        
        if model_name in models:
            # Collect the three accuracy types to plot their histograms side-by-side
            model_acc_vals_q.append(models[model_name]['avg_ref_acc'])
            model_acc_vals_qC.append(models[model_name]['avg_giveaway_acc'])
            model_acc_vals_qAfC.append(models[model_name]['avg_giveaway_acc_afc'])

    # After looping over datasets, plot all three distributions as side-by-side histograms
    data_to_plot = [model_acc_vals_q, model_acc_vals_qC, model_acc_vals_qAfC]
    labels = ["Question", "Question + Context", "Question + Answer-Free Context"]
    colors = [plot_colors[0], plot_colors[1], plot_colors[2]]

    bins = np.linspace(0, 1, 16)
    bar_width = (bins[1] - bins[0]) / 4  # Make bars beside, not stacked or overlapping
    rwidth = bar_width * 16

    
    # Plot a smoothed PDF for each set of accuracy values
    from scipy.stats import gaussian_kde

    x_grid = np.linspace(0, 1, 200)
    for i, vals in enumerate(data_to_plot):
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            pdf = kde(x_grid)
            plt.plot(
                x_grid,
                pdf,
                label=labels[i],
                color=colors[i],
                alpha=0.8,
                linewidth=3,
            )
            plt.fill_between(x_grid, pdf, color=colors[i], alpha=0.2)
        elif len(vals) == 1:
            # Draw a spike since kde needs > 1 point
            plt.axvline(vals[0], color=colors[i], label=labels[i], alpha=0.85, linewidth=2)
        # else, skip if empty
    plt.gca().set_yticks([])

    plt.legend(loc='upper left')
    plt.title('Impact of Context on Benchmark Accuracy')
    plt.xlabel('Benchmark Accuracy Distribution \n Over All Datasets (Gaussian KDE)')
    plt.ylabel('')  # Remove units/label from the y-axis
    # plt.ylabel('Density')
    plt.tight_layout()
    
    
    # Save the plot
    os.makedirs(f'./imgs/impact_of_context/q_vs_qAFC', exist_ok=True)
    plt.savefig(f'./imgs/impact_of_context/q_vs_qAFC/{model_name}.svg', dpi=300, bbox_inches='tight')
    plt.close()

print(f"Scatterplots saved for {len(all_models)} models")




