"""Plot accuracy comparison: rewritten question vs question with answer-free context."""

import os
import copy
import numpy as np

import json







from plot_config import COLORS as plot_colors, MARKERS as plot_markers
for generating_model_name in ['gpt120b']:  #, 'gpt20b', 'Q235B'
    impact_of_giveaway = dict()
    for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc','data-subset-500-afc']:
    # for dataset_fldr in ['data-post-cutoff-afc','data-subset-500-afc']:
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
                d_fp = d_fp.replace('mmajursk','mmajurski')
                # d_fp = d_fp.replace('/home/mmajursk/github/lm-rewrite-uplift','/Users/mmajursk/github/lm-rewrite-uplift')
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
            avg_uplift_per_model_dataset[dataset_name][model_name] = {}
            if len(score_lists['ref_accs']) > 0:
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_ref_acc'] = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_ref_acc_giveaway'] = sum(score_lists['ref_accs_giveaway']) / len(score_lists['ref_accs_giveaway'])
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_rewrite_acc'] = sum(score_lists['rewrite_accs']) / len(score_lists['rewrite_accs'])
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_rewrite_acc_giveaway'] = sum(score_lists['rewrite_accs_giveaway']) / len(score_lists['rewrite_accs_giveaway'])
            if len(score_lists['ref_accs_afc']) > 0:
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_ref_acc_afc'] = sum(score_lists['ref_accs_afc']) / len(score_lists['ref_accs_afc'])
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_ref_acc_giveaway_afc'] = sum(score_lists['ref_accs_giveaway_afc']) / len(score_lists['ref_accs_giveaway_afc'])
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_rewrite_acc_afc'] = sum(score_lists['rewrite_accs_afc']) / len(score_lists['rewrite_accs_afc'])
                avg_uplift_per_model_dataset[dataset_name][model_name]['avg_rewrite_acc_giveaway_afc'] = sum(score_lists['rewrite_accs_giveaway_afc']) / len(score_lists['rewrite_accs_giveaway_afc'])


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




    post_cutoff_datasets = ['ai_plan', 'ai_plan_yb','arXiv_2502_17521v1', 'arXiv_2502_17521v1_yb', 'hle']
    all_datasets_ordered = [d for d in post_cutoff_datasets if d in all_datasets] + [d for d in all_datasets if d not in post_cutoff_datasets]
    all_datasets = all_datasets_ordered



    # Create scatterplots per model
    import matplotlib.pyplot as plt
    import numpy as np



    plt.figure(figsize=(7, 7))
    figure_has_content = False

    # Create a scatterplot for each model
    for m_idx, model_name in enumerate(all_models):            
        
        # Plot data points for each dataset
        for d_idx, dataset_name in enumerate(all_datasets):
            if not dataset_name in avg_uplift_per_model_dataset.keys():
                continue
            models = avg_uplift_per_model_dataset[dataset_name]                    

            if model_name in models:
                ref_acc = models[model_name]['avg_ref_acc_giveaway_afc']
                rewrite_acc = models[model_name]['avg_rewrite_acc_afc']
                
                marker = plot_markers[d_idx % len(plot_markers)]
                color = plot_colors[m_idx % len(plot_colors)]
                plt.scatter(ref_acc,rewrite_acc, 
                        marker=marker, color=color, 
                        s=80, 
                        alpha=1.0, 
                        label=f'{dataset_name}')
                figure_has_content = True

        
    # Add diagonal line (y=x) representing no change
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # Add grid and labels
    # plt.grid(True, alpha=0.3)

    plt.xlabel('Benchmark Accuracy for Original Question + Answer-Free Context')
    plt.ylabel('Benchmark Accuracy for Rewrite of Answer-Free Context')
    plt.title(f'Benchmark Accuracy Comparing Question + Answer-Free Context to\nRewrite of Question')

    # plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)


    # Make the plot square to maintain aspect ratio
    plt.axis('equal')
    ax = plt.gca()

    all_models = [m.split('/')[1] if '/' in m else m for m in all_models]

    # Create legend for colors (datasets)
    handles_color = [plt.Line2D([0], [0], color=plot_colors[i], lw=5, alpha=1.0) for i, _ in enumerate(all_models)]
    labels_color = [f"{m}" for m in all_models]
    # legend1 = ax.legend(handles_color, labels_color, title="Evaluation Models", loc="upper left", fontsize='small')  #x-small
    legend1 = ax.legend(handles_color, labels_color, title="Evaluation Models", loc="upper left", fontsize='x-small')  #x-small

    # Create legend for markers (models)
    handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color='black', linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(all_datasets)]
    labels_marker = [f"{d}" for d in all_datasets]
    legend2 = ax.legend(handles_marker, labels_marker, title="Datasets", loc="lower right", fontsize='x-small') # x-small

    # Add both legends
    ax.add_artist(legend1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # Save the plot
    os.makedirs(f'./imgs/rQ_vs_qAFC/', exist_ok=True)
    plt.savefig(f'./imgs/rQ_vs_qAFC/{generating_model_name}-orig-rewrite.svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scatterplots saved for {len(all_models)} models")






