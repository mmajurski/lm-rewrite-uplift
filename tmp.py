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



response = """<output_format>\nQuestion: The United States\u2019 AI Action Plan assigns several federal agencies the responsibility of eliminating regulatory obstacles that impede artificial\u2011intelligence innovation. Identify each of these primary agencies and describe one concrete step that each agency is directed to take to remove such barriers.\n\nExplanation: The answer lists the agencies named in the plan\u2014OSTP, OMB, FCC, and FTC\u2014and provides the specific action each is tasked with (e.g., OSTP\u2019s request for information, OMB\u2019s review and repeal of hindering regulations, FCC\u2019s assessment of state AI rules under the Communications Act, and FTC\u2019s review of investigations and orders to avoid undue burdens on AI).\n\nCorrect Answer:\n- Office of Science and Technology Policy (OSTP) \u2013 launch a Request for Information from businesses and the public about federal regulations that hinder AI innovation and work with relevant agencies to address them.\n- Office of Management and Budget (OMB) \u2013 work with all federal agencies to identify, revise, or repeal regulations, rules, memoranda, guidance, and agreements that unnecessarily impede AI development or deployment.\n- Federal Communications Commission (FCC) \u2013 evaluate whether state AI regulations interfere with the FCC\u2019s obligations and authority under the Communications Act of 1934.\n- Federal Trade Commission (FTC) \u2013 review ongoing investigations, final orders, consent decrees, and injunctions to ensure they do not unduly burden AI innovation and, where appropriate, seek to modify or set aside such actions.\n</output_format>"""
answer_parser.parse_generated_open(response)



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
