import os
import sys
import json
import numpy as np
import glob

import answer_parser


# ifp = '/home/mmajursk/Downloads/cite_datasets/'

# # Loop over all JSON files in the directory
# for json_file in glob.glob(os.path.join(ifp, '*.json')):
#     print(f"Processing: {json_file}")
    
#     try:
#         # Load the JSON data
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # Re-save with proper indentation
#         with open(json_file, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
        
#         print(f"Successfully reformatted: {json_file}")
        
#     except Exception as e:
#         print(f"Error processing {json_file}: {e}")

# print("All JSON files processed!")



# response = """<output_format>\nQuestion: The United States\u2019 AI Action Plan assigns several federal agencies the responsibility of eliminating regulatory obstacles that impede artificial\u2011intelligence innovation. Identify each of these primary agencies and describe one concrete step that each agency is directed to take to remove such barriers.\n\nExplanation: The answer lists the agencies named in the plan\u2014OSTP, OMB, FCC, and FTC\u2014and provides the specific action each is tasked with (e.g., OSTP\u2019s request for information, OMB\u2019s review and repeal of hindering regulations, FCC\u2019s assessment of state AI rules under the Communications Act, and FTC\u2019s review of investigations and orders to avoid undue burdens on AI).\n\nCorrect Answer:\n- Office of Science and Technology Policy (OSTP) \u2013 launch a Request for Information from businesses and the public about federal regulations that hinder AI innovation and work with relevant agencies to address them.\n- Office of Management and Budget (OMB) \u2013 work with all federal agencies to identify, revise, or repeal regulations, rules, memoranda, guidance, and agreements that unnecessarily impede AI development or deployment.\n- Federal Communications Commission (FCC) \u2013 evaluate whether state AI regulations interfere with the FCC\u2019s obligations and authority under the Communications Act of 1934.\n- Federal Trade Commission (FTC) \u2013 review ongoing investigations, final orders, consent decrees, and injunctions to ensure they do not unduly burden AI innovation and, where appropriate, seek to modify or set aside such actions.\n</output_format>"""
# answer_parser.parse_generated_open(response)



# ifp = '/home/mmajursk/github/lm-rewrite-uplift/data-post-cutoff/source_data/arXiv_2502_17521v1_novel.json'
# with open(ifp, 'r') as f:
#     data = json.load(f)

# for d in data:
#     keys_to_keep = {'context', 'question', 'answer'}
#     for key in list(d.keys()):
#         if key not in keys_to_keep:
#             del d[key]

# # Save the cleaned data back to the same file
# ifp = ifp.replace(".json", "-sub.json")
# with open(ifp, 'w') as f_out:
#     json.dump(data, f_out, indent=2)



# fp = '/home/mmajursk/github/lm-rewrite-uplift/data-subset-500/logs-oe-Q235B-filtered-orig'
# import os
# import json
# fns = [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith('.json')]
# fns.sort()
# models_dict = dict()
# models = set()
# datasets = set()


# for fn in fns:
#     with open(fn, 'r') as f:
#         data = json.load(f)

#     model_name = data['eval']['model']
#     if model_name == 'v_llm/Llama-4-Maverick-17B-128E-Instruct-FP8':
#         os.remove(fn)
    
# fn = '/home/mmajursk/github/lm-rewrite-uplift/data-post-cutoff/oe-gpt120b-filtered/ai_plan.json'
# with open(fn, 'r') as f:
#     data = json.load(f)

# for sample in data:
#     keys_to_keep = {'question', 'answer', 'orig_question', 'orig_answer'}
#     for key in list(sample.keys()):
#         if key not in keys_to_keep:
#             del sample[key]

# with open('ai_plan-debug.json', 'w') as f:
#     json.dump(data, f, indent=2)


# fn = '/home/mmajursk/github/lm-rewrite-uplift/data-post-cutoff/oe-gpt120b-filtered/ai_plan_yb.json'
# with open(fn, 'r') as f:
#     data = json.load(f)

# for sample in data:
#     keys_to_keep = {'question', 'answer', 'orig_question', 'orig_answer'}
#     for key in list(sample.keys()):
#         if key not in keys_to_keep:
#             del sample[key]

# with open('ai_plan_yb-debug.json', 'w') as f:
#     json.dump(data, f, indent=2)








# ifp = '/home/mmajursk/github/lm-rewrite-uplift/data-subset-500-SU/logs-oe-gpt120b-filtered-reformat'
# fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json')]
# fns.sort()

# for fn in fns:
#     fn = os.path.join(ifp, fn)
#     with open(fn, 'r') as f:
#         data = json.load(f)

#     data['eval']['task_args']['dataset_fldr'] = data['eval']['task_args']['dataset_fldr'].replace('data-subset-500-SU2', 'data-subset-500-SU')
#     data['eval']['task_args']['dataset_fldr'] = data['eval']['task_args']['dataset_fldr'].replace('mmajurski', 'mmajursk')

#     # Overwrite the JSON file to reflect the new fields (i.e., save the updated data)
#     with open(fn, 'w') as f:
#         json.dump(data, f, indent=2)







# data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-subset-500-SU'
# # data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-post-cutoff'
# fldrs = [fn for fn in os.listdir(data_dir) if fn.startswith('logs-')]
# fldrs.sort()

# for fldr in fldrs:
#     ifp = os.path.join(data_dir, fldr)
#     to_delete = []

#     fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json')]
#     fns.sort()

#     for fn in fns:
#         fn = os.path.join(ifp, fn)
#         with open(fn, 'r') as f:
#             data = json.load(f)

#         if 'gemma-3-270m' in data['eval']['model']:
#             to_delete.append(fn)
#         if 'Qwen3-0.6B' in data['eval']['model']:
#             to_delete.append(fn)

#     for fn in to_delete:
#         os.remove(os.path.join(ifp, fn))

#     print(f"Deleted {len(to_delete)} files from {ifp}")




# data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-subset-500'
data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-subset-500-afc'
# data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-post-cutoff'
# data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-post-cutoff-afc'
# data_dir = '/home/mmajursk/github/lm-rewrite-uplift/data-subset-500-SU'
fldrs = ['source_data'] #, 'oe-gpt120b', 'oe-gpt120b-filtered' ['oe-gpt120b', 'oe-gpt120b-filtered', 'oe-Q235B', 'oe-Q235B-filtered']

for fldr in fldrs:
    ifp = os.path.join(data_dir, fldr)
    
    fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json')]
    fns.sort()
    print(fns)

    for fn in fns:
        fn = os.path.join(ifp, fn)
        
        with open(fn, 'r') as f:
            data = json.load(f)

        # if 'reformat_question' in data[0]:
        #     continue
        print(f"Processing {fn}")

        for i in range(len(data)):
            del_keys = ['reformat_response','reformat_scratchpad', 'explanation', 'reformat_answer_giveaway_response', 'reformat_answer_giveaway_scratchpad', 'orig_answer_giveaway_response', 'orig_answer_giveaway_scratchpad']
            del_keys.extend(['reformat_answer_giveaway_score', 'reformat_question_similarity_score', 'reformat_answer_similarity_score', 'reformat_question_clarity_score', 'reformat_question_difficulty_score', 'reformat_question_groundedness_score'])
            for k in list(data[i].keys()):
                if '_embeddings' in k:
                    del_keys.append(k)
            for key in del_keys:
                if key in data[i]:
                    del data[i][key]
            if 'answer' not in data[i]:
                data[i]['answer'] = data[i]['orig_answer']
            # data[i]['reformat_question'] = data[i]['question']
            # data[i]['reformat_answer'] = data[i]['answer']
            # del data[i]['afc_question']
            # del data[i]['afc_answer']
            # 
            if 'question' in data[i]:
                del data[i]['question']
            if 'answer' in data[i]:
                del data[i]['answer']

            # data[i]['context'] = data[i]['context_no_answer']
            # del data[i]['context_no_answer']


            key_order = ['context', 'orig_question', 'orig_answer', 'reformat_question', 'reformat_answer', 'reformat_question', 'reformat_answer']
            # Build new ordered dict
            new_d = {}
            for k in key_order:
                if k in data[i]:
                    new_d[k] = data[i][k]
            # Add remaining keys not in key_order
            for k in data[i]:
                if k not in key_order:
                    new_d[k] = data[i][k]
            data[i] = new_d
        

        with open(fn, 'w') as f:
            json.dump(data, f, indent=2)








