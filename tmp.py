import os
import sys
import json
import numpy as np
import glob


ifp = '/home/mmajursk/Downloads/cite_datasets/'

# Loop over all JSON files in the directory
for json_file in glob.glob(os.path.join(ifp, '*.json')):
    print(f"Processing: {json_file}")
    
    try:
        # Load the JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Re-save with proper indentation
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully reformatted: {json_file}")
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

print("All JSON files processed!")
