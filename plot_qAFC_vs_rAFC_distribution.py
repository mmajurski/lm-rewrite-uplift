
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


impact_of_giveaway = dict()
for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc','data-subset-500-afc']:
    print(f"Processing {dataset_fldr}")
    for generating_model_name in ['gpt120b']:
        
        main_dir = f'./{dataset_fldr}/'

        if dataset_fldr.endswith('-afc'):
            context_type = 'afc'
            context_str = '-afc'
        else:
            context_type = 'orig'
            context_str = ""
        
        qAFC_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig-giveaway')
        rAFC_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat')  # logs-oe-gpt120b-afc-filtered-reformat
        rAFC_giveaway_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat-giveaway')
        # qAFC_giveaway_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig-giveaway')

        if os.path.exists(qAFC_folder):
            qAFC_logs = [fn for fn in os.listdir(qAFC_folder) if fn.endswith('.json')]
        else:
            qAFC_logs = []
        if os.path.exists(rAFC_folder):
            rAFC_logs = [fn for fn in os.listdir(rAFC_folder) if fn.endswith('.json')]
        else:
            rAFC_logs = []
        if os.path.exists(rAFC_giveaway_folder):
            rAFC_giveaway_logs = [fn for fn in os.listdir(rAFC_giveaway_folder) if fn.endswith('.json')]
        else:
            rAFC_giveaway_logs = []


        for log_type in ['ref', 'rewrite', 'giveaway']:
            if log_type == 'ref':
                logs = qAFC_logs
                question_folder = qAFC_folder
            elif log_type == 'rewrite':
                logs = rAFC_logs
                question_folder = rAFC_folder
            elif log_type == 'giveaway':
                logs = rAFC_giveaway_logs
                question_folder = rAFC_giveaway_folder
            else:
                raise ValueError(f"Invalid log type: {log_type}")


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
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc'] = acc  # Q+C
                            elif log_type == 'rewrite':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['rewrite_acc'] = acc  # R
                            else:
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['giveaway_acc'] = acc  # R+C
                        else:
                            if log_type == 'ref':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc_afc'] = acc
                            elif log_type == 'rewrite':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['rewrite_acc_afc'] = acc
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
                model_scores[model_name] = {'ref_accs': [], 'giveaway_accs': [], 'rewrite_accs': [], 'ref_accs_afc': [], 'giveaway_accs_afc': [], 'rewrite_accs_afc': []}
            if 'ref_acc' in scores and 'giveaway_acc' in scores and 'rewrite_acc' in scores:
                model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                model_scores[model_name]['giveaway_accs'].append(scores['giveaway_acc'])
                model_scores[model_name]['rewrite_accs'].append(scores['rewrite_acc'])
            if 'ref_acc_afc' in scores and 'giveaway_acc_afc' in scores and 'rewrite_acc_afc' in scores:
                model_scores[model_name]['ref_accs_afc'].append(scores['ref_acc_afc'])
                model_scores[model_name]['giveaway_accs_afc'].append(scores['giveaway_acc_afc'])
                model_scores[model_name]['rewrite_accs_afc'].append(scores['rewrite_acc_afc'])
    
    # Compute averages for each model
    for model_name, score_lists in model_scores.items():
        if len(score_lists['ref_accs']) > 0:
            avg_ref_acc = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
            avg_giveaway_acc = sum(score_lists['giveaway_accs']) / len(score_lists['giveaway_accs'])
            avg_rewrite_acc = sum(score_lists['rewrite_accs']) / len(score_lists['rewrite_accs'])
            avg_ref_acc_afc = sum(score_lists['ref_accs_afc']) / len(score_lists['ref_accs_afc'])
            avg_giveaway_acc_afc = sum(score_lists['giveaway_accs_afc']) / len(score_lists['giveaway_accs_afc'])
            avg_rewrite_acc_afc = sum(score_lists['rewrite_accs_afc']) / len(score_lists['rewrite_accs_afc'])
            avg_uplift_per_model_dataset[dataset_name][model_name] = {
                'avg_ref_acc': avg_ref_acc,
                'avg_giveaway_acc': avg_giveaway_acc,
                'avg_rewrite_acc': avg_rewrite_acc,
                'avg_ref_acc_afc': avg_ref_acc_afc,
                'avg_giveaway_acc_afc': avg_giveaway_acc_afc,
                'avg_rewrite_acc_afc': avg_rewrite_acc_afc,
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

    
    model_acc_vals_q = []
    model_acc_vals_r = []
    model_acc_vals_r_giveaway = []
    model_acc_vals_q_afc = []
    model_acc_vals_r_afc = []
    model_acc_vals_r_giveaway_afc = []

    
    # Plot data points for each dataset
    for d_idx, dataset_name in enumerate(all_datasets):
        if not dataset_name in avg_uplift_per_model_dataset.keys():
            continue
        models = avg_uplift_per_model_dataset[dataset_name]                    

        
        if model_name in models:
            # Collect the three accuracy types to plot their histograms side-by-side
            model_acc_vals_q.append(models[model_name]['avg_ref_acc'])  # Q+C
            model_acc_vals_r.append(models[model_name]['avg_rewrite_acc']) # R
            model_acc_vals_r_giveaway.append(models[model_name]['avg_giveaway_acc']) # R+C

            model_acc_vals_q_afc.append(models[model_name]['avg_ref_acc_afc']) # Q+AFC
            model_acc_vals_r_afc.append(models[model_name]['avg_rewrite_acc_afc']) # R
            model_acc_vals_r_giveaway_afc.append(models[model_name]['avg_giveaway_acc_afc']) # R+AFC
            

    # After looping over datasets, plot all three distributions as side-by-side histograms
    data_to_plot = [model_acc_vals_q, model_acc_vals_r]#, model_acc_vals_r_giveaway]
    labels = ["Question + Context", "Rewritten Question"]#, "Rewritten Question + Context"]
    colors = [plot_colors[0], plot_colors[1], plot_colors[2]]

    bins = np.linspace(0, 1, 16)
    bar_width = (bins[1] - bins[0]) / 4  # Make bars beside, not stacked or overlapping
    rwidth = bar_width * 16

    plt.figure(figsize=(4, 2.5))    
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
    os.makedirs(f'./imgs/impact_of_context/qC_vs_rC', exist_ok=True)
    plt.savefig(f'./imgs/impact_of_context/qC_vs_rC/{model_name}.svg', dpi=300, bbox_inches='tight')
    plt.close()


    # After looping over datasets, plot all three distributions as side-by-side histograms
    data_to_plot = [model_acc_vals_q_afc, model_acc_vals_r_afc]#, model_acc_vals_r_giveaway_afc]
    labels = ["Question + Answer-Free Context", "Rewritten Question"]#, "Rewritten Question + Answer-Free Context"]
    colors = [plot_colors[0], plot_colors[1], plot_colors[2]]

    bins = np.linspace(0, 1, 16)
    bar_width = (bins[1] - bins[0]) / 4  # Make bars beside, not stacked or overlapping
    rwidth = bar_width * 16

    plt.figure(figsize=(4, 2.5))    
    # Plot a smoothed PDF for each set of accuracy values
    

    x_grid = np.linspace(0, 1, 200)
    mv_Y  = 0.0
    for i, vals in enumerate(data_to_plot):
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            pdf = kde(x_grid)
            mv_Y = max(mv_Y, np.max(pdf))
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
    # Add 10% padding to the top of the plot to prevent legend overlap
    plt.ylim(0, mv_Y * 1.8)

    plt.legend(loc='upper left')
    plt.title('Impact of Context on Benchmark Accuracy')
    plt.xlabel('Benchmark Accuracy Distribution \n Over All Datasets (Gaussian KDE)')
    plt.ylabel('')  # Remove units/label from the y-axis
    # plt.ylabel('Density')
    plt.tight_layout()
    
    
    # Save the plot
    os.makedirs(f'./imgs/impact_of_context/qAFC_vs_rAFC', exist_ok=True)
    plt.savefig(f'./imgs/impact_of_context/qAFC_vs_rAFC/{model_name}.svg', dpi=300, bbox_inches='tight')
    plt.close()

print(f"Scatterplots saved for {len(all_models)} models")




