
import os
import copy
import numpy as np

import json


plot_colors = [
    '#1f77b4',  # blue
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
    '#ff7f0e',  # orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
    # Additional colors
    '#393b79',  # dark blue
    '#637939',  # dark green
    '#8c6d31',  # dark brown
    '#843c39',  # dark red
    '#7b4173',  # dark purple
    '#17becf',  # cyan (repeat for more variety)
    '#bc80bd',  # lavender
    '#ffed6f',  # yellow
    '#1b9e77',  # teal
    '#e7298a',  # magenta
    '#66a61e',  # olive green
    '#e6ab02',  # mustard
    '#a6761d',  # brown
    '#666666',  # dark gray
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

# ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc', 'data-subset-500-afc']
for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc', 'data-subset-500-afc']:
    for question_source in ['orig', 'reformat']:
        for generating_model_name in ['gpt120b', 'Q235B']:
            
            main_dir = f'./{dataset_fldr}/'

            if dataset_fldr.endswith('-afc'):
                context_type = 'afc'
                context_str = '-afc'
            else:
                context_type = 'orig'
                context_str = ""
            
            question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-{question_source}')
            giveaway_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-{question_source}-giveaway')

            if os.path.exists(question_folder):
                ref_logs = [fn for fn in os.listdir(question_folder) if fn.endswith('.json')]
            else:
                ref_logs = []
            if os.path.exists(giveaway_question_folder):
                giveaway_logs = [fn for fn in os.listdir(giveaway_question_folder) if fn.endswith('.json')]
            else:
                giveaway_logs = []


            impact_of_giveaway = dict()
            # top level key is the dataset name
            # second level key is the question text
            # then a set of property keys: 
            # context: the grounding context
            # correct_answer:  the correct answer
            # model_name_metrics:  ['ref_acc', 'reformat_acc', 'acc_uplift'] with the associated accuracy values
            for log_type in ['ref', 'giveaway']:
                if log_type == 'ref':
                    logs = ref_logs
                    question_folder = question_folder
                else:
                    logs = giveaway_logs
                    question_folder = giveaway_question_folder


                for fn_idx, fn in enumerate(logs):
                    print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
                    with open(os.path.join(question_folder, fn), 'r') as f:
                        data = json.load(f)
                    
                    model_name = data['eval']['model'].replace("v_llm/", "")
                    model_name = model_name.split('/')[1] if '/' in model_name else model_name
                    dataset_name = data['eval']['task_registry_name']

                    d_fp = data['eval']['task_args']['dataset_fldr']
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
                            if log_type == 'ref':
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['ref_acc'] = acc    
                            else:
                                impact_of_giveaway[dataset_name][orig_question]['models'][model_name]['giveaway_acc'] = acc    
                                


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
                            model_scores[model_name] = {'ref_accs': [], 'giveaway_accs': []}
                        if 'ref_acc' in scores and 'giveaway_acc' in scores:
                            model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                            model_scores[model_name]['giveaway_accs'].append(scores['giveaway_acc'])
                
                # Compute averages for each model
                for model_name, score_lists in model_scores.items():
                    if len(score_lists['ref_accs']) > 0:
                        avg_ref_acc = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
                        avg_giveaway_acc = sum(score_lists['giveaway_accs']) / len(score_lists['giveaway_accs'])
                        
                        avg_uplift_per_model_dataset[dataset_name][model_name] = {
                            'avg_ref_acc': avg_ref_acc,
                            'avg_giveaway_acc': avg_giveaway_acc,
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
            plt.figure(figsize=(8, 8))
            figure_has_content = False

            # Create a scatterplot for each model
            for m_idx, model_name in enumerate(all_models):            
                
                # Plot data points for each dataset
                for d_idx, dataset_name in enumerate(all_datasets):
                    if not dataset_name in avg_uplift_per_model_dataset.keys():
                        continue
                    models = avg_uplift_per_model_dataset[dataset_name]                    

                    if model_name in models:
                        ref_acc = models[model_name]['avg_ref_acc']
                        giveaway_acc = models[model_name]['avg_giveaway_acc']
                        num_questions = models[model_name]['num_questions']
                        
                        marker = plot_markers[d_idx % len(plot_markers)]
                        color = plot_colors[m_idx % len(plot_colors)]
                        plt.scatter(ref_acc, giveaway_acc, 
                                marker=marker, color=color, 
                                s=80, 
                                alpha=1.0, 
                                label=f'{dataset_name} (n={num_questions})')
                    figure_has_content = True

            if not figure_has_content:
                continue
                
            # Add diagonal line (y=x) representing no change
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            # Add grid and labels
            # plt.grid(True, alpha=0.3)
            if question_source == 'orig':
                plt.title(f'Average Reference vs Giveaway Accuracy')
                plt.xlabel('Average Dataset Reference Accuracy')
            else:
                plt.title(f'Average Reformat vs Giveaway Accuracy')
                plt.xlabel('Average Dataset Reformat Accuracy')
            plt.ylabel('Average Dataset Giveaway Accuracy')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
            legend1 = ax.legend(handles_color, labels_color, title="Evaluation Models", loc="center right", fontsize='x-small')  #x-small

            # Create legend for markers (models)
            handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color='black', linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(all_datasets)]
            labels_marker = [f"{d}" for d in all_datasets]
            legend2 = ax.legend(handles_marker, labels_marker, title="Datasets", loc="lower right", fontsize='x-small') # x-small

            # Add both legends
            ax.add_artist(legend1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            
            # Save the plot
            os.makedirs(f'./imgs/{dataset_fldr}', exist_ok=True)
            plt.savefig(f'./imgs/{dataset_fldr}/{generating_model_name}-{question_source}-giveaway.svg', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Scatterplots saved for {len(all_models)} models")




