"""Plot answer giveaway score vs accuracy (binned by accuracy) for question variants."""
import sys, os

from plot_config import COLORS as plot_colors, MARKERS as plot_markers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import os
import copy
import numpy as np

import json
import re

import model_interface

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()




valid_model_list = ['gpt-oss-20b', 'gpt-oss-120b', 'gemma-3-4b-it', 'Qwen3-1.7B']
# valid_model_list = ['gpt-oss-120b']

#for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-subset-500-SU', 'data-post-cutoff-afc', 'data-subset-500-afc']:
for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-subset-500-SU', 'data-post-cutoff-afc','data-subset-500-afc']:
    for generating_model_name in ['gpt120b', 'gpt20b', 'Q235B']:
        main_dir = f'./{dataset_fldr}/'

        if dataset_fldr.endswith('-afc'):
                context_type = 'afc'
                context_str = '-afc'
        else:
            context_type = 'orig'
            context_str = ""
            
        ref_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig')
        reformat_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat')
        

        if os.path.exists(ref_question_folder):
            ref_logs = [fn for fn in os.listdir(ref_question_folder) if fn.endswith('.json')]
        else:
            ref_logs = []
        if os.path.exists(reformat_question_folder):
            reformat_logs = [fn for fn in os.listdir(reformat_question_folder) if fn.endswith('.json')]
        else:
            reformat_logs = []

        if len(ref_logs) == 0 and len(reformat_logs) == 0:
            continue


    
        impact_of_reformat = dict()
        # top level key is the dataset name
        # second level key is the question text
        # then a set of property keys: 
        # context: the grounding context
        # correct_answer:  the correct answer
        # model_name_metrics:  ['ref_acc', 'reformat_acc', 'acc_uplift'] with the associated accuracy values
        for log_type in ['ref', 'reformat']:
            if log_type == 'ref':
                logs = ref_logs
                question_folder = ref_question_folder
            else:
                logs = reformat_logs
                question_folder = reformat_question_folder



            for fn_idx, fn in enumerate(logs):
                print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
                with open(os.path.join(question_folder, fn), 'r') as f:
                    data = json.load(f)
                
                model_name = data['eval']['model'].replace("v_llm/", "")
                model_name = model_name.split('/')[1] if '/' in model_name else model_name
                
                if model_name not in valid_model_list:
                    continue
                dataset_name = data['eval']['task_registry_name']

                d_fp = data['eval']['task_args']['dataset_fldr']
                d_fp = d_fp.replace('mmajursk','mmajurski')
                with open(d_fp, 'r') as f:
                    source_dataset = json.load(f)

                if dataset_name not in impact_of_reformat:
                    impact_of_reformat[dataset_name] = dict()
                
                samples = data['samples']
                try:
                    for q_id, sample in enumerate(samples):
                        orig_question = source_dataset[q_id]['orig_question']
                        # reformat_question = source_dataset[q_id]['question']
                        # # reformat_answer = source_dataset[q_id]['answer']
                        # orig_answer = source_dataset[q_id]['orig_answer']
                        # context = source_dataset[q_id]['context']
                        if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
                            acc = sample['scores']['model_graded_qa']['value'] == 'C'
                            if orig_question not in impact_of_reformat[dataset_name]:
                                impact_of_reformat[dataset_name][orig_question] = dict()
                                # impact_of_reformat[dataset_name][orig_question]['orig_question'] = orig_question
                                # impact_of_reformat[dataset_name][orig_question]['reformat_question'] = reformat_question
                                # impact_of_reformat[dataset_name][orig_question]['orig_answer'] = orig_answer

                                impact_of_reformat[dataset_name][orig_question]['orig_answer_giveaway_score'] = source_dataset[q_id]['orig_answer_giveaway_score']
                                impact_of_reformat[dataset_name][orig_question]['reformat_answer_giveaway_score'] = source_dataset[q_id]['reformat_answer_giveaway_score']

                                
                                # impact_of_reformat[dataset_name][orig_question]['reformat_answer'] = reformat_answer
                                # impact_of_reformat[dataset_name][orig_question]['context'] = context
                                impact_of_reformat[dataset_name][orig_question]['models'] = dict()
                            if model_name not in impact_of_reformat[dataset_name][orig_question]['models']:
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name] = dict()
                            if log_type == 'ref':
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name]['ref_acc'] = acc    
                            else:
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name]['reformat_acc'] = acc    
                except:
                    print(f"Error processing {fn}")
                    print(f"model_name {model_name}")
                    print(f"source_dataset {d_fp}")
                    raise Exception("Error processing sample")
                        



        # Compute binned data per model per dataset
        binned_plot_data = dict()

        for dataset_name, questions in impact_of_reformat.items():
            if dataset_name not in binned_plot_data:
                binned_plot_data[dataset_name] = dict()
            
            # Collect data per model for this dataset
            model_scores = dict()
            for question, sub_data in questions.items():
                orig_answer_giveaway_scores = sub_data['orig_answer_giveaway_score']
                reformat_answer_giveaway_scores = sub_data['reformat_answer_giveaway_score']
                
                models = sub_data['models']
                for model_name, scores in models.items():
                    if model_name not in model_scores:
                        model_scores[model_name] = {'ref_accs': [], 'reformat_accs': [], 'orig_answer_giveaway_scores': [], 'reformat_answer_giveaway_scores': []}
                    if 'ref_acc' in scores and 'reformat_acc' in scores:
                        model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                        model_scores[model_name]['reformat_accs'].append(scores['reformat_acc'])
                        model_scores[model_name]['orig_answer_giveaway_scores'].append(orig_answer_giveaway_scores)
                        model_scores[model_name]['reformat_answer_giveaway_scores'].append(reformat_answer_giveaway_scores)

            # Create binned data for each model in this dataset
            for model_name, score_lists in model_scores.items():
                if len(score_lists['ref_accs']) == 0:
                    continue
                    
                # Convert to numpy arrays for easier sorting
                ref_accs = np.array(score_lists['ref_accs'])
                reformat_accs = np.array(score_lists['reformat_accs'])
                orig_giveaway_scores = np.array(score_lists['orig_answer_giveaway_scores'])
                reformat_giveaway_scores = np.array(score_lists['reformat_answer_giveaway_scores'])
                
                # Sort by accuracy and create bins
                bin_size = 20
                
                # For reference accuracy plots
                #ref_sorted_indices = np.argsort(ref_accs)
                ref_sorted_indices = np.random.permutation(len(ref_accs))
                
                ref_accs_sorted = ref_accs[ref_sorted_indices]
                orig_giveaway_sorted = orig_giveaway_scores[ref_sorted_indices]
                
                # For reformat accuracy plots  
                # reformat_sorted_indices = np.argsort(reformat_accs)
                reformat_sorted_indices = np.random.permutation(len(reformat_accs))
                reformat_accs_sorted = reformat_accs[reformat_sorted_indices]
                reformat_giveaway_sorted = reformat_giveaway_scores[reformat_sorted_indices]
                
                # Create bins
                ref_bins = []
                reformat_bins = []
                
                # Reference accuracy bins
                for i in range(0, len(ref_accs_sorted), bin_size):
                    end_idx = min(i + bin_size, len(ref_accs_sorted))
                    if end_idx > i:  # Only create bin if it has at least one element
                        ref_bins.append({
                            'avg_acc': np.mean(ref_accs_sorted[i:end_idx]),
                            'avg_giveaway': np.mean(orig_giveaway_sorted[i:end_idx]),
                            'num_questions': end_idx - i
                        })
                
                # Reformat accuracy bins
                for i in range(0, len(reformat_accs_sorted), bin_size):
                    end_idx = min(i + bin_size, len(reformat_accs_sorted))
                    if end_idx > i:  # Only create bin if it has at least one element
                        reformat_bins.append({
                            'avg_acc': np.mean(reformat_accs_sorted[i:end_idx]),
                            'avg_giveaway': np.mean(reformat_giveaway_sorted[i:end_idx]),
                            'num_questions': end_idx - i
                        })
                
                if model_name not in binned_plot_data[dataset_name]:
                    binned_plot_data[dataset_name][model_name] = {
                        'ref_bins': ref_bins,
                        'reformat_bins': reformat_bins
                    }

        # # Save the average uplift data
        # with open(f'avg_uplift_per_model_dataset_{generating_model_name}{giveaway_name}.json', 'w') as f:
        #     json.dump(avg_uplift_per_model_dataset, f, indent=2)


        # Create scatterplots per model
        import matplotlib.pyplot as plt
        import numpy as np



        # Get unique models and datasets from binned data
        all_models = set()
        all_datasets = set()
        for dataset_name, models in binned_plot_data.items():
            all_datasets.add(dataset_name)
            all_models.update(models.keys())

        all_datasets = list(all_datasets)
        all_datasets.sort()
        all_models = list(all_models)
        all_models.sort()
        
        

        for reformat_flag in [True, False]:
            

            # Plot data points for each dataset
            for d_idx, dataset_name in enumerate(all_datasets):
                plt.figure(figsize=(8, 8))
                figure_has_content = False

                all_x_values = []
                all_y_values = []

                # Create a scatterplot for each model
                for m_idx, model_name in enumerate(all_models):
                
                    if dataset_name not in binned_plot_data or model_name not in binned_plot_data[dataset_name]:
                        continue
                        
                    model_data = binned_plot_data[dataset_name][model_name]
                    
                    if reformat_flag:
                        bins = model_data['reformat_bins']
                    else:
                        bins = model_data['ref_bins']

                    cur_acc_data = [b['avg_acc'] for b in bins]
                    cur_giveaway_data = [b['avg_giveaway'] for b in bins]

                    all_x_values.extend(cur_acc_data)
                    all_y_values.extend(cur_giveaway_data)

                    #marker = plot_markers[d_idx % len(plot_markers)]
                    marker = plot_markers[0]
                    color = plot_colors[m_idx % len(plot_colors)]
                    # color = plot_colors[d_idx % len(plot_colors)]
                    plt.scatter(cur_acc_data, cur_giveaway_data, 
                            marker=marker, color=color, 
                            s=80, 
                            alpha=1.0, 
                            label=f'{dataset_name}')
                    figure_has_content = True
                        
                        
                if not figure_has_content:
                    continue
                
                # Add grid and labels
                plt.grid(True, alpha=0.3)
                if reformat_flag:
                    plt.xlabel('Average Reformat Accuracy (Binned)')
                    plt.ylabel('Average Reformat Answer Giveaway (Binned)')
                else:
                    plt.xlabel('Average Reference Accuracy (Binned)')
                    plt.ylabel('Average Reference Answer Giveaway (Binned)')
                plt.title(f'Answer Giveaway vs Accuracy (Binned by Accuracy)')
                
                # Make the plot square to maintain aspect ratio
                # plt.axis('equal')
                plt.xlim(0, 1)
                plt.ylim(np.percentile(all_y_values, 1) - 0.1, np.percentile(all_y_values, 99) + 0.1)
                # plt.xlim(0, 1)
                # plt.ylim(0, 1)

                ax = plt.gca()

                all_models_labels = [m.split('/')[1] if '/' in m else m for m in all_models]

                # Create legend for colors (models)
                handles_color = [plt.Line2D([0], [0], color=plot_colors[i], lw=5, alpha=1.0) for i, _ in enumerate(all_models_labels)]
                labels_color = [f"{m}" for m in all_models_labels]
                legend1 = ax.legend(handles_color, labels_color, title="Evaluation Models", loc="upper left", fontsize='x-small')  #x-small

                # Create legend for markers (datasets)
                cur_datasets = [dataset_name]
                handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color='black', linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(cur_datasets)]
                # labels_marker = [f"{d}" for d in all_datasets]
                labels_marker = [f"{d}" for d in cur_datasets]

                legend2 = ax.legend(handles_marker, labels_marker, title="Datasets", loc="lower right", fontsize='x-small') # x-small

                # Add both legends
                ax.add_artist(legend1)
                plt.tight_layout()
                
                # Save the plot
                os.makedirs(f'./imgs/{dataset_fldr}', exist_ok=True)
                post_script = 'reformat' if reformat_flag else 'orig'
                plt.savefig(f'./imgs/{dataset_fldr}/answer_giveaway_{generating_model_name}_{post_script}_{dataset_name}{context_str}.svg', dpi=300, bbox_inches='tight')
                plt.close()

            print(f"Scatterplots saved for {len(all_models)} models")




