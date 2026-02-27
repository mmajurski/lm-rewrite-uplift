"""Plot HLE (Humanity Last Exam) benchmark results across question rewrite variants."""
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




model_name_translator = {
    'gpt120b': 'gpt-oss-120b',
    'Q235B': 'Qwen3-30B-A3B-Instruct-2507',
    'gpt20b': 'gpt-oss-20b',
}

for dataset_fldr in ['data-post-cutoff','data-post-cutoff-afc']:
    for generating_model_name in ['gpt120b', 'Q235B', 'gpt20b']:
            main_dir = f'./{dataset_fldr}/'
            
            if dataset_fldr.endswith('-afc'):
                context_type = 'afc'
                context_str = '-afc'
            else:
                context_type = 'orig'
                context_str = ""
                
            # main_dir = './data-post-cutoff/'
            ref_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig')
            reformat_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat')
            giveaway_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig-giveaway')

            if os.path.exists(ref_question_folder):
                ref_logs = [fn for fn in os.listdir(ref_question_folder) if fn.endswith('.json')]
            else:
                ref_logs = []
            if os.path.exists(reformat_question_folder):
                reformat_logs = [fn for fn in os.listdir(reformat_question_folder) if fn.endswith('.json')]
            else:
                reformat_logs = []
            if os.path.exists(giveaway_question_folder):
                giveaway_logs = [fn for fn in os.listdir(giveaway_question_folder) if fn.endswith('.json')]
            else:
                giveaway_logs = []


        
            impact_of_reformat = dict()
            # top level key is the dataset name
            # second level key is the question text
            # then a set of property keys: 
            # context: the grounding context
            # correct_answer:  the correct answer
            # model_name_metrics:  ['ref_acc', 'reformat_acc', 'acc_uplift'] with the associated accuracy values
            for log_type in ['ref', 'reformat', 'giveaway']:
                if log_type == 'ref':
                    logs = ref_logs
                    question_folder = ref_question_folder
                elif log_type == 'reformat':
                    logs = reformat_logs
                    question_folder = reformat_question_folder
                elif log_type == 'giveaway':
                    logs = giveaway_logs
                    question_folder = giveaway_question_folder
                else:
                    raise ValueError(f"Invalid log type: {log_type}")

                logs = [fn for fn in logs if 'hle' in fn]

                for fn_idx, fn in enumerate(logs):
                    # print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
                    with open(os.path.join(question_folder, fn), 'r') as f:
                        data = json.load(f)
                    
                    model_name = data['eval']['model'].replace("v_llm/", "")
                    model_name = model_name.split('/')[1] if '/' in model_name else model_name
                    dataset_name = data['eval']['task_registry_name']

                    d_fp = data['eval']['task_args']['dataset_fldr']
                    # d_fp = d_fp.replace('mmajursk','mmajurski')
                    with open(d_fp, 'r') as f:
                        source_dataset = json.load(f)

                    if dataset_name not in impact_of_reformat:
                        impact_of_reformat[dataset_name] = dict()
                    
                    samples = data['samples']
                    for q_id, sample in enumerate(samples):
                        question = sample['input']
                        orig_question = source_dataset[q_id]['orig_question']
                        reformat_question = source_dataset[q_id]['reformat_question']
                        # reformat_answer = source_dataset[q_id]['answer']
                        orig_answer = source_dataset[q_id]['orig_answer']
                        context = source_dataset[q_id]['context']
                        if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
                            acc = sample['scores']['model_graded_qa']['value'] == 'C'
                            if orig_question not in impact_of_reformat[dataset_name]:
                                impact_of_reformat[dataset_name][orig_question] = dict()
                                impact_of_reformat[dataset_name][orig_question]['orig_question'] = orig_question
                                impact_of_reformat[dataset_name][orig_question]['reformat_question'] = reformat_question
                                impact_of_reformat[dataset_name][orig_question]['orig_answer'] = orig_answer
                                # impact_of_reformat[dataset_name][orig_question]['reformat_answer'] = reformat_answer
                                impact_of_reformat[dataset_name][orig_question]['context'] = context
                                impact_of_reformat[dataset_name][orig_question]['models'] = dict()
                            if model_name not in impact_of_reformat[dataset_name][orig_question]['models']:
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name] = dict()
                            if log_type == 'ref':
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name]['ref_acc'] = acc    
                            elif log_type == 'reformat':
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name]['reformat_acc'] = acc    
                            elif log_type == 'giveaway':
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name]['giveaway_acc'] = acc    
                            else:
                                raise ValueError(f"Invalid log type: {log_type}")


            # Compute average acc_uplift per model and dataset
            avg_uplift_per_model_dataset = dict()

            for dataset_name, questions in impact_of_reformat.items():
                if dataset_name not in avg_uplift_per_model_dataset:
                    avg_uplift_per_model_dataset[dataset_name] = dict()
                
                # Group questions by model to compute averages
                model_scores = dict()
                for question, sub_data in questions.items():
                    models = sub_data['models']
                    for model_name, scores in models.items():
                        if model_name not in model_scores:
                            model_scores[model_name] = {'uplifts': [], 'ref_accs': [], 'reformat_accs': [], 'giveaway_accs': []}
                        if 'ref_acc' in scores and 'reformat_acc' in scores:
                            model_scores[model_name]['uplifts'].append(float(scores['reformat_acc']) - float(scores['ref_acc']))
                            model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                            model_scores[model_name]['reformat_accs'].append(scores['reformat_acc'])
                            if 'giveaway_acc' in scores:
                                model_scores[model_name]['giveaway_accs'].append(scores['giveaway_acc'])
                            else:
                                model_scores[model_name]['giveaway_accs'].append(np.nan)

                # Compute averages for each model
                for model_name, score_lists in model_scores.items():
                    if len(score_lists['uplifts']) > 0:
                        avg_uplift = sum(score_lists['uplifts']) / len(score_lists['uplifts'])
                        avg_ref_acc = sum(score_lists['ref_accs']) / len(score_lists['ref_accs'])
                        avg_reformat_acc = sum(score_lists['reformat_accs']) / len(score_lists['reformat_accs'])
                        giveaway_accs = [a for a in score_lists['giveaway_accs'] if not np.isnan(a)]
                        if len(giveaway_accs) > 0:
                            avg_giveaway_acc = sum(giveaway_accs) / len(giveaway_accs)
                        else:
                            avg_giveaway_acc = np.nan
                        
                        avg_uplift_per_model_dataset[dataset_name][model_name] = {
                            'avg_acc_uplift': avg_uplift,
                            'avg_ref_acc': avg_ref_acc,
                            'avg_reformat_acc': avg_reformat_acc,
                            'avg_giveaway_acc': avg_giveaway_acc,
                            'num_questions': len(score_lists['uplifts'])
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
            plt.figure(figsize=(4.5, 4.5))
            figure_has_content = False
            all_y_values = []
            all_x_values = []

            # Create a scatterplot for each model
            for m_idx, model_name in enumerate(all_models):            
                
                # Plot data points for each dataset
                for d_idx, dataset_name in enumerate(all_datasets):
                    if not dataset_name in avg_uplift_per_model_dataset.keys():
                        continue
                    models = avg_uplift_per_model_dataset[dataset_name]

                    if model_name in models:
                        ref_acc = models[model_name]['avg_ref_acc']
                        reformat_acc = models[model_name]['avg_reformat_acc']
                        num_questions = models[model_name]['num_questions']
                        all_x_values.append(ref_acc)
                        all_y_values.append(reformat_acc)
                        marker = plot_markers[m_idx % len(plot_markers)]
                        color = plot_colors[m_idx % len(plot_colors)]
                        plt.scatter(ref_acc, reformat_acc, 
                                marker=marker, color=color, 
                                s=120, 
                                alpha=1.0, 
                                label=f'{dataset_name} (n={num_questions})')
                        figure_has_content = True

            if figure_has_content:
                # Add diagonal line (y=x) representing no change
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

                model_name = model_name_translator[generating_model_name]
                if context_type == 'afc':
                    plt.xlabel('Benchmark Accuracy for Original Questions (no Context)')
                    plt.ylabel('Benchmark Accuracy for Rewritten Questions (no Context)')
                    plt.title(f'HLE Benchmark Subset Rewritten by {model_name}\nGrounded with AnswerFree Context')
                else:
                    plt.xlabel('Benchmark Accuracy for Original Questions (no Context)')
                    plt.ylabel('Benchmark Accuracy for Rewritten Questions (no Context)')
                    plt.title(f'HLE Benchmark Subset Rewritten by {model_name}\nGrounded with Context')
                
                
                # Make the plot square to maintain aspect ratio
                plt.axis('equal')
                ax = plt.gca()

                all_models = [m.split('/')[1] if '/' in m else m for m in all_models]

                # Create legend for (marker, color) combinations to clearly delineate (model, dataset) pairs
                # Only show a subset of models in the legend; see appendix for full legend
                legend_models_subset = {'gpt-5', 'gpt-5-mini', 'gpt-oss-120b', 'Llama-4-Maverick-Instruct', 'gemma-3-27b-it', 'Qwen3-235B-Instruct-2507'}
                handles = []
                labels = []
                for m_idx, model in enumerate(all_models):
                    marker = plot_markers[m_idx % len(plot_markers)]
                    color = plot_colors[m_idx % len(plot_colors)]
                    model_name = model
                    model_name = f"{model_name.replace('-FP8','')}"
                    model_name = f"{model_name.replace('-17B-128E','')}"
                    model_name = f"{model_name.replace('-A3B','')}"
                    model_name = f"{model_name.replace('-A22B','')}"
                    if model_name not in legend_models_subset:
                        continue

                    handles.append(
                        plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', markersize=8, label=model_name)
                    )
                    labels.append(f"{model_name}")
                legend1 = ax.legend(handles, labels, title="Evaluation Models", loc="lower right", fontsize='small', ncol=1)

                # Add both legends
                ax.add_artist(legend1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                mv = np.max([np.max(all_y_values), np.max(all_x_values)]) + 0.05
                plt.xlim(0, mv) #0.55)
                plt.ylim(0, mv) #0.55)

                # Save the plot
                os.makedirs(f'./imgs/{dataset_fldr}', exist_ok=True)
                cur_ofp = f'./imgs/{dataset_fldr}/{generating_model_name}{context_str}_hle.svg'
                print(cur_ofp)
                plt.savefig(cur_ofp, dpi=300, bbox_inches='tight')
                plt.close()





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
            plt.figure(figsize=(4.5, 4.5))
            figure_has_content = False
            all_y_values = []
            all_x_values = []

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
                        if np.isnan(giveaway_acc):
                            continue
                        num_questions = models[model_name]['num_questions']
                        all_x_values.append(ref_acc)
                        all_y_values.append(giveaway_acc)
                        marker = plot_markers[m_idx % len(plot_markers)]
                        color = plot_colors[m_idx % len(plot_colors)]
                        plt.scatter(ref_acc, giveaway_acc, 
                                marker=marker, color=color, 
                                s=120, 
                                alpha=1.0, 
                                label=f'{dataset_name} (n={num_questions})')
                        figure_has_content = True

            if figure_has_content:
                # Add diagonal line (y=x) representing no change
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                model_name = model_name_translator[generating_model_name]
                
                if context_type == 'afc':
                    plt.xlabel('Benchmark Accuracy for Original Questions (no Context)')
                    plt.ylabel('Benchmark Accuracy for Original Questions (with Answer Free Context)')
                    plt.title(f'HLE Benchmark Subset Rewritten by {model_name}\nGrounded with Answer-Free Context')
                else:
                    plt.xlabel('Benchmark Accuracy for Original Questions (no Context)')
                    plt.ylabel('Benchmark Accuracy for Original Questions (with Context)')
                    plt.title(f'HLE Benchmark SubsetRewritten by {model_name}\nGrounded with Context')
                
                
                # Make the plot square to maintain aspect ratio
                plt.axis('equal')
                ax = plt.gca()

                all_models = [m.split('/')[1] if '/' in m else m for m in all_models]

                # Create legend for (marker, color) combinations to clearly delineate (model, dataset) pairs
                # Only show a subset of models in the legend; see appendix for full legend
                legend_models_subset = {'gpt-5', 'gpt-5-mini', 'gpt-oss-120b', 'Llama-4-Maverick-Instruct', 'gemma-3-27b-it', 'Qwen3-235B-Instruct-2507'}
                handles = []
                labels = []
                for m_idx, model in enumerate(all_models):
                    
                    marker = plot_markers[m_idx % len(plot_markers)]
                    color = plot_colors[m_idx % len(plot_colors)]
                    model_name = model
                    model_name = f"{model_name.replace('-FP8','')}"
                    model_name = f"{model_name.replace('-17B-128E','')}"
                    model_name = f"{model_name.replace('-A3B','')}"
                    model_name = f"{model_name.replace('-A22B','')}"
                    if model_name not in legend_models_subset:
                        continue
                    handles.append(
                        plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', markersize=8, label=model_name)
                    )
                    labels.append(f"{model_name}")
                legend1 = ax.legend(handles, labels, title="Evaluation Models", loc="lower right", fontsize='small', ncol=1)

                # Add both legends
                ax.add_artist(legend1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                mv = np.max([np.max(all_y_values), np.max(all_x_values)]) + 0.05
                plt.xlim(0, mv) #0.55)
                plt.ylim(0, mv) #0.55)

                # Save the plot
                os.makedirs(f'./imgs/{dataset_fldr}', exist_ok=True)
                cur_ofp = f'./imgs/{dataset_fldr}/{generating_model_name}{context_str}_hle_giveaway.svg'
                print(cur_ofp)
                plt.savefig(cur_ofp, dpi=300, bbox_inches='tight')
                plt.close()