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



response = """
<document_analysis>\n1. **Thoughtful Content Examination**  \n   - The context provides two films:  \n     * *Secret Things* \u2013 a 2002 French erotic thriller directed by Jean\u2011Claude Brisseau, who died on 11\u202fMay\u202f2019.  \n     * *All I Desire* \u2013 a 1953 American romantic drama directed by Douglas Sirk, who died on 14\u202fJanuary\u202f1987.  \n   - The original question asks which of the two films \u201chas the director died later.\u201d The phrase \u201cdied later\u201d is ambiguous without explicit dates.\n\n2. **Concept Exploration**  \n   - Implicit assumption: \u201cdied later\u201d refers to a more recent death (i.e., a later chronological point).  \n   - The test taker must compare the two death dates and decide which director\u2019s death occurred more recently.\n\n3. **Intentional Question Planning**  \n   - The rewritten question should present the two films, their release years, directors, and death dates.  \n   - It should then ask which film is linked to the director whose death happened more recently.  \n   - The question must be self\u2011contained, allowing a domain expert to answer without consulting the original context.\n\n4. **Detailed Assumption Expansion**  \n   - Assumption: the test taker knows how to interpret dates (i.e., 2019 is later than 1987).  \n   - Assumption: the test taker understands that \u201cmore recently\u201d and \u201clater\u201d are synonymous in this temporal comparison.  \n   - The question includes all necessary background (film titles, release years, director names, death dates) so that no external reference is needed.\n\n5. **Giving Away the Answer**  \n   - The question avoids phrasing that hints at the answer (e.g., it does not say \u201cthe director who died in 2019\u201d).  \n   - It uses neutral language: \u201cWhich of these two films is associated with the director whose death occurred more recently?\u201d  \n   - This phrasing forces the test taker to perform the date comparison rather than infer the answer from a cue.\n\nOverall, the rewritten question will be clear, unambiguous, and fully self\u2011contained while preserving the original semantic intent.\n</document_analysis>\n\n<output_format>\nQuestion: Two films are described as follows:\n\n- *Secret Things* is a 2002 French erotic thriller directed by Jean\u2011Claude Brisseau, who died on 11\u202fMay\u202f2019.\n- *All I Desire* is a 1953 American romantic drama directed by Douglas Sirk, who died on 14\u202fJanuary\u202f1987.\n\nWhich of these two films is associated with the director whose death occurred more recently?\n\nExplanation: The director of *Secret Things* (Jean\u2011Claude Brisseau) died in 2019, which is later (more recent) than the director of *All I Desire* (Douglas Sirk), who died in 1987. Therefore, the film linked to the more recent death is *Secret Things*.\n\nCorrect Answer: Secret Things\n</output_format>"""
answer_parser.parse_generated_open(response)