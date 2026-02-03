
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


embedding_models = ['Qwen3-Embedding-8B', 'e5-mistral-7b-instruct']


# for dataset_fldr in ['data-post-cutoff','data-subset-500', 'data-post-cutoff-afc','data-subset-500-afc']:
#     for generating_model_name in ['gpt120b', 'gpt20b', 'Q235B']:
#         main_dir = f'./{dataset_fldr}/'

#         if dataset_fldr.endswith('-afc'):
#             context_type = 'afc'
#             context_str = '-afc'
#         else:
#             context_type = 'orig'
#             context_str = ""
            
#         ref_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-orig')
#         reformat_question_folder = os.path.join(main_dir, f'logs-oe-{generating_model_name}{context_str}-filtered-reformat')

#         if os.path.exists(ref_question_folder):
#             ref_logs = [fn for fn in os.listdir(ref_question_folder) if fn.endswith('.json')]
#         else:
#             ref_logs = []
#         if os.path.exists(reformat_question_folder):
#             reformat_logs = [fn for fn in os.listdir(reformat_question_folder) if fn.endswith('.json')]
#         else:
#             reformat_logs = []


    
#         impact_of_reformat = dict()
#         for log_type in ['ref', 'reformat']:
#             if log_type == 'ref':
#                 logs = ref_logs
#                 question_folder = ref_question_folder
#             else:
#                 logs = reformat_logs
#                 question_folder = reformat_question_folder



#             for fn_idx, fn in enumerate(logs):
#                 # print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
#                 with open(os.path.join(question_folder, fn), 'r') as f:
#                     data = json.load(f)
                
#                 model_name = data['eval']['model'].replace("v_llm/", "")
#                 model_name = model_name.split('/')[1] if '/' in model_name else model_name
#                 dataset_name = data['eval']['task_registry_name']

#                 d_fp = data['eval']['task_args']['dataset_fldr']
#                 # d_fp = d_fp.replace('mmajursk','mmajurski')
#                 d_fp = d_fp.replace('/home/mmajursk/github/lm-rewrite-uplift','/Users/mmajursk/github/lm-rewrite-uplift')
#                 with open(d_fp, 'r') as f:
#                     source_dataset = json.load(f)

#                 if dataset_name not in impact_of_reformat:
#                     impact_of_reformat[dataset_name] = dict()
                
#                 samples = data['samples']
#                 try:
#                     for q_id, sample in enumerate(samples):
#                         question = sample['input']
#                         orig_question = source_dataset[q_id]['orig_question']
#                         reformat_question = source_dataset[q_id]['reformat_question']

#                         # reformat_answer = source_dataset[q_id]['answer']
#                         orig_answer = source_dataset[q_id]['orig_answer']
#                         context = source_dataset[q_id]['context']
#                         if 'model_graded_qa' in sample['scores']:  # if the question graded correctly
#                             acc = sample['scores']['model_graded_qa']['value'] == 'C'
#                             if orig_question not in impact_of_reformat[dataset_name]:
#                                 impact_of_reformat[dataset_name][orig_question] = dict()
#                                 impact_of_reformat[dataset_name][orig_question]['orig_question'] = orig_question
#                                 impact_of_reformat[dataset_name][orig_question]['reformat_question'] = reformat_question
#                                 impact_of_reformat[dataset_name][orig_question]['orig_answer'] = orig_answer
#                                 # impact_of_reformat[dataset_name][orig_question]['reformat_answer'] = reformat_answer
#                                 impact_of_reformat[dataset_name][orig_question]['context'] = context
#                                 impact_of_reformat[dataset_name][orig_question]['models'] = dict()
#                             if model_name not in impact_of_reformat[dataset_name][orig_question]['models']:
#                                 impact_of_reformat[dataset_name][orig_question]['models'][model_name] = dict()
#                             if log_type == 'ref':
#                                 impact_of_reformat[dataset_name][orig_question]['models'][model_name]['ref_acc'] = acc    
#                             else:
#                                 impact_of_reformat[dataset_name][orig_question]['models'][model_name]['reformat_acc'] = acc    

#                             if 'embeddings' not in impact_of_reformat[dataset_name][orig_question].keys():
#                                 impact_of_reformat[dataset_name][orig_question]['embeddings'] = dict()
#                             for e_model in embedding_models:
#                                 if log_type == 'ref':
#                                     impact_of_reformat[dataset_name][orig_question]['embeddings'][f'{e_model}_cosine_embR_embC'] = source_dataset[q_id][f'{e_model}_embeddings']['cosine_embR_embC']
#                                 else:
#                                     impact_of_reformat[dataset_name][orig_question]['embeddings'][f'{e_model}_cosine_embO_embC'] = source_dataset[q_id][f'{e_model}_embeddings']['cosine_embO_embC']
#                 except Exception as e:
#                     print(f"Error processing {fn}")
#                     print(f"model_name {model_name}")
#                     print(f"source_dataset {d_fp}")
#                     raise Exception("Error processing sample")



#         per_dataset_scores = dict()
#         for dataset_name, questions in impact_of_reformat.items():
            
#             # Group questions by model to compute averages
#             model_scores = dict()
#             for question, sub_data in questions.items():
#                 models = sub_data['models']
#                 embeddings = sub_data['embeddings']
#                 for model_name, scores in models.items():
#                     if model_name not in model_scores:
#                         model_scores[model_name] = {'ref_accs': [], 'reformat_accs': []}
#                         for e_model in embedding_models:
#                             model_scores[model_name][f'{e_model}_cosine_embR_embC'] = []
#                             model_scores[model_name][f'{e_model}_cosine_embO_embC'] = []
#                     if 'ref_acc' in scores and 'reformat_acc' in scores:
#                         model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
#                         model_scores[model_name]['reformat_accs'].append(scores['reformat_acc'])
#                         for e_model in embedding_models:
#                             model_scores[model_name][f'{e_model}_cosine_embR_embC'].append(float(embeddings[f'{e_model}_cosine_embR_embC']))
#                             model_scores[model_name][f'{e_model}_cosine_embO_embC'].append(float(embeddings[f'{e_model}_cosine_embO_embC']))
#             per_dataset_scores[dataset_name] = model_scores
                



#         # Create scatterplots per model
#         import matplotlib.pyplot as plt
#         import numpy as np



#         # Get unique models across all datasets
#         all_models = set()
#         all_datasets = set()
#         for dataset_name, models in per_dataset_scores.items():
#             all_models.update(models.keys())
#             all_datasets.add(dataset_name)
        

#         all_datasets = list(all_datasets)
#         all_datasets.sort()
#         all_models = list(all_models)
#         all_models.sort()
        

#         # Create a scatterplot for each model
#         for embedding_model in embedding_models:
#             plt.figure(figsize=(6, 6))
#             figure_has_content = False
#             for m_idx, model_name in enumerate(all_models):            
                
#                 # Plot data points for each dataset
#                 for d_idx, dataset_name in enumerate(all_datasets):
#                     if not dataset_name in per_dataset_scores.keys():
#                         continue
#                     models = per_dataset_scores[dataset_name]

#                     if model_name in models:
#                         ref_acc = np.asarray(models[model_name]['ref_accs']).astype(float)
#                         reformat_acc = np.asarray(models[model_name]['reformat_accs']).astype(float)
#                         cosine_embR_embC = np.asarray(models[model_name][f'{embedding_model}_cosine_embR_embC']).astype(float)
#                         cosine_embO_embC = np.asarray(models[model_name][f'{embedding_model}_cosine_embO_embC']).astype(float)
                        
#                         delta_acc = (reformat_acc - ref_acc)
#                         delta_emb = (cosine_embR_embC - cosine_embO_embC)
#                         # delta_acc = delta_acc / np.mean(reformat_acc)
#                         # delta_emb = delta_emb / np.mean(cosine_embR_embC)

#                         delta_acc = np.mean(delta_acc)
#                         delta_emb = np.mean(delta_emb)
#                         marker = plot_markers[d_idx % len(plot_markers)]
#                         # color = plot_colors[m_idx % len(plot_colors)]
#                         color = plot_colors[d_idx % len(plot_markers)]

#                         plt.scatter(np.mean(delta_acc), np.mean(delta_emb), 
#                                 marker=marker, color=color, 
#                                 s=80, 
#                                 alpha=1.0, 
#                                 label=f'{dataset_name}')
#                         figure_has_content = True

#             if not figure_has_content:
#                 continue
                
#             # Add grid and labels
#             # plt.grid(True, alpha=0.3)
#             plt.xlabel('Delta Benchmark Accuracy (Rewrite_Q - Orig_Q)') # \nlarger values indicate better Rewrite_Q performance
#             plt.ylabel('Delta Embedding Cosine Similarity\nCosineSim(Rewrite_Q, Context) - CosineSim(Orig_Q, Context)') # \nlarger values indicate better alignment between question and context
#             # plt.xlabel('Dataset Percentage Accuracy Change (Reformat - Orig)/Reformat')
#             # plt.ylabel('Dataset Percentage Embedding Cosine Change (embR_embC - embO_embC)/embR_embC')
#             plt.title(f'Delta Benchmark Accuracy vs Delta Embedding Cosine Similarity\n({embedding_model})')
#             # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             # plt.legend()
#             # plt.xlim(0, 1)
#             # plt.ylim(0, 1)
#             # Only apply if current min xlim is greater than -0.1
#             # curr_xlim = plt.xlim()
#             # if curr_xlim[0] > -0.1:
#             #     plt.xlim(left=-0.1)
#             # curr_ylim = plt.ylim()
#             # if curr_xlim[0] > -0.02:
#             #     plt.ylim(bottom=-0.02)
            
#             # Make the plot square to maintain aspect ratio
#             # plt.axis('equal')
#             # Draw a vertical line at x=0 and a horizontal line at y=0
#             plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
#             plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
#             ax = plt.gca()

#             all_models = [m.split('/')[1] if '/' in m else m for m in all_models]

#             # # Create legend for colors (datasets)
#             # handles_color = [plt.Line2D([0], [0], color=plot_colors[i], lw=5, alpha=1.0) for i, _ in enumerate(all_models)]
#             # labels_color = [f"{m}" for m in all_models]
#             # legend1 = ax.legend(handles_color, labels_color, title="Evaluation Models", loc="upper right", fontsize='x-small')  #x-small

#             # Create legend for markers (models)
#             # handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color='black', linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(all_datasets)]
#             handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color=plot_colors[i], linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(all_datasets)]
#             labels_marker = [f"{d}" for d in all_datasets]
#             legend2 = ax.legend(handles_marker, labels_marker, title="Datasets", loc="lower right", fontsize='x-small') # x-small

#             # Add both legends
#             # ax.add_artist(legend1)
#             plt.grid(True, alpha=0.3)
#             plt.tight_layout()
            
#             # Save the plot
#             os.makedirs(f'./imgs/{dataset_fldr}', exist_ok=True)
#             cur_ofp = f'./imgs/{dataset_fldr}/acc_vs_embedding_{generating_model_name}_{embedding_model}.svg'
#             print(cur_ofp)
#             plt.savefig(cur_ofp, dpi=300, bbox_inches='tight')
#             plt.close()

#             print(f"Scatterplots saved for {len(all_models)} models")










for generating_model_name in ['gpt120b', 'gpt20b', 'Q235B']:
    impact_of_reformat = dict()
    for dataset_fldr in ['data-post-cutoff-afc','data-subset-500-afc']:
    
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
        
        for log_type in ['ref', 'reformat']:
            if log_type == 'ref':
                logs = ref_logs
                question_folder = ref_question_folder
            else:
                logs = reformat_logs
                question_folder = reformat_question_folder



            for fn_idx, fn in enumerate(logs):
                # print(f"Processing {fn} ({fn_idx+1}/{len(logs)})")
                with open(os.path.join(question_folder, fn), 'r') as f:
                    data = json.load(f)
                
                model_name = data['eval']['model'].replace("v_llm/", "")
                model_name = model_name.split('/')[1] if '/' in model_name else model_name
                dataset_name = data['eval']['task_registry_name']

                d_fp = data['eval']['task_args']['dataset_fldr']
                # d_fp = d_fp.replace('mmajursk','mmajurski')
                # d_fp = d_fp.replace('/home/mmajursk/github/lm-rewrite-uplift','/Users/mmajursk/github/lm-rewrite-uplift')
                with open(d_fp, 'r') as f:
                    source_dataset = json.load(f)

                if dataset_name not in impact_of_reformat:
                    impact_of_reformat[dataset_name] = dict()
                
                samples = data['samples']
                try:
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
                            else:
                                impact_of_reformat[dataset_name][orig_question]['models'][model_name]['reformat_acc'] = acc    

                            if 'embeddings' not in impact_of_reformat[dataset_name][orig_question].keys():
                                impact_of_reformat[dataset_name][orig_question]['embeddings'] = dict()
                            for e_model in embedding_models:
                                if log_type == 'ref':
                                    impact_of_reformat[dataset_name][orig_question]['embeddings'][f'{e_model}_cosine_embR_embC'] = source_dataset[q_id][f'{e_model}_embeddings']['cosine_embR_embC']
                                else:
                                    impact_of_reformat[dataset_name][orig_question]['embeddings'][f'{e_model}_cosine_embO_embC'] = source_dataset[q_id][f'{e_model}_embeddings']['cosine_embO_embC']
                except Exception as e:
                    print(f"Error processing {fn}")
                    print(f"model_name {model_name}")
                    print(f"source_dataset {d_fp}")
                    raise Exception("Error processing sample")



    per_dataset_scores = dict()
    for dataset_name, questions in impact_of_reformat.items():
        
        # Group questions by model to compute averages
        model_scores = dict()
        for question, sub_data in questions.items():
            models = sub_data['models']
            embeddings = sub_data['embeddings']
            for model_name, scores in models.items():
                if model_name not in model_scores:
                    model_scores[model_name] = {'ref_accs': [], 'reformat_accs': []}
                    for e_model in embedding_models:
                        model_scores[model_name][f'{e_model}_cosine_embR_embC'] = []
                        model_scores[model_name][f'{e_model}_cosine_embO_embC'] = []
                if 'ref_acc' in scores and 'reformat_acc' in scores:
                    model_scores[model_name]['ref_accs'].append(scores['ref_acc'])
                    model_scores[model_name]['reformat_accs'].append(scores['reformat_acc'])
                    for e_model in embedding_models:
                        model_scores[model_name][f'{e_model}_cosine_embR_embC'].append(float(embeddings[f'{e_model}_cosine_embR_embC']))
                        model_scores[model_name][f'{e_model}_cosine_embO_embC'].append(float(embeddings[f'{e_model}_cosine_embO_embC']))
        per_dataset_scores[dataset_name] = model_scores
            



    # Create scatterplots per model
    import matplotlib.pyplot as plt
    import numpy as np



    # Get unique models across all datasets
    all_models = set()
    all_datasets = set()
    for dataset_name, models in per_dataset_scores.items():
        all_models.update(models.keys())
        all_datasets.add(dataset_name)
    

    all_datasets = list(all_datasets)
    all_datasets.sort()
    all_models = list(all_models)
    all_models.sort()

    post_cutoff_datasets = ['ai_plan', 'ai_plan_yb','arXiv_2502_17521v1', 'arXiv_2502_17521v1_yb', 'hle']
    all_datasets_ordered = [d for d in post_cutoff_datasets if d in all_datasets] + [d for d in all_datasets if d not in post_cutoff_datasets]
    all_datasets = all_datasets_ordered
    

    # Create a scatterplot for each model
    
    for embedding_model in embedding_models:
        total_count = 0
        ur_quad_count = 0
        plt.figure(figsize=(6, 6))
        figure_has_content = False
        for m_idx, model_name in enumerate(all_models):            
            
            # Plot data points for each dataset
            for d_idx, dataset_name in enumerate(all_datasets):
                if not dataset_name in per_dataset_scores.keys():
                    continue
                models = per_dataset_scores[dataset_name]

                if model_name in models:
                    ref_acc = np.asarray(models[model_name]['ref_accs']).astype(float)
                    reformat_acc = np.asarray(models[model_name]['reformat_accs']).astype(float)
                    cosine_embR_embC = np.asarray(models[model_name][f'{embedding_model}_cosine_embR_embC']).astype(float)
                    cosine_embO_embC = np.asarray(models[model_name][f'{embedding_model}_cosine_embO_embC']).astype(float)
                    
                    delta_acc = (reformat_acc - ref_acc)
                    delta_emb = (cosine_embR_embC - cosine_embO_embC)
                    # delta_acc = delta_acc / np.mean(reformat_acc)
                    # delta_emb = delta_emb / np.mean(cosine_embR_embC)

                    delta_acc = np.mean(delta_acc)
                    delta_emb = np.mean(delta_emb)
                    marker = plot_markers[d_idx % len(plot_markers)]
                    # color = plot_colors[m_idx % len(plot_colors)]
                    color = plot_colors[d_idx % len(plot_markers)]

                    total_count += int(delta_acc.size)
                    m1 = delta_acc > 0
                    m2 = delta_emb > 0
                    m = np.logical_and(m1, m2)
                    ur_quad_count += int(np.sum(m))

                    plt.scatter(np.mean(delta_acc), np.mean(delta_emb), 
                            marker=marker, color=color, 
                            s=80, 
                            alpha=1.0, 
                            label=f'{dataset_name}')
                    figure_has_content = True

        if not figure_has_content:
            continue

        print(f'Embedding Model {embedding_model} as {total_count} points, of which {ur_quad_count} are in UR quad')
            
        # Add grid and labels
        # plt.grid(True, alpha=0.3)
        plt.xlabel('Delta Benchmark Accuracy (Rewrite_Q - Orig_Q)') # \nlarger values indicate better Rewrite_Q performance
        plt.ylabel('Delta Embedding Cosine Similarity\nCosineSim(Rewrite_Q, Context) - CosineSim(Orig_Q, Context)') # \nlarger values indicate better alignment between question and context
        # plt.xlabel('Dataset Percentage Accuracy Change (Reformat - Orig)/Reformat')
        # plt.ylabel('Dataset Percentage Embedding Cosine Change (embR_embC - embO_embC)/embR_embC')
        plt.title(f'Delta Benchmark Accuracy vs Delta Embedding Cosine Similarity\n({embedding_model})')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.legend()
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # Only apply if current min xlim is greater than -0.1
        # curr_xlim = plt.xlim()
        # if curr_xlim[0] > -0.1:
        #     plt.xlim(left=-0.1)
        # curr_ylim = plt.ylim()
        # if curr_xlim[0] > -0.02:
        #     plt.ylim(bottom=-0.02)
        
        # Make the plot square to maintain aspect ratio
        # plt.axis('equal')
        # Draw a vertical line at x=0 and a horizontal line at y=0
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax = plt.gca()

        all_models = [m.split('/')[1] if '/' in m else m for m in all_models]

        # # Create legend for colors (datasets)
        # handles_color = [plt.Line2D([0], [0], color=plot_colors[i], lw=5, alpha=1.0) for i, _ in enumerate(all_models)]
        # labels_color = [f"{m}" for m in all_models]
        # legend1 = ax.legend(handles_color, labels_color, title="Evaluation Models", loc="upper right", fontsize='x-small')  #x-small

        # Create legend for markers (models)
        # handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color='black', linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(all_datasets)]
        handles_marker = [plt.Line2D([0], [0], marker=plot_markers[i], color=plot_colors[i], linestyle='None', markersize=6, alpha=1.0) for i, _ in enumerate(all_datasets)]
        labels_marker = [f"{d}" for d in all_datasets]
        legend2 = ax.legend(handles_marker, labels_marker, title="Datasets", loc="upper right", fontsize='x-small') # x-small

        # Add both legends
        # ax.add_artist(legend1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(f'./imgs/emb', exist_ok=True)
        cur_ofp = f'./imgs/emb/acc_vs_embedding_{generating_model_name}_{embedding_model}.svg'
        print(cur_ofp)
        plt.savefig(cur_ofp, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatterplots saved for {len(all_models)} models")

