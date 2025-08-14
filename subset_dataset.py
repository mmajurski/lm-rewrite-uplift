import os
import json
import random

ifp = './source_data'
N = 500
ofp = f'./data-subset-{N}/source_data'
os.makedirs(ofp, exist_ok=True)

fns = [fn for fn in os.listdir(ifp) if fn.endswith('.json')]


# Process each merged file
for fn in fns:
    print(f"Processing {fn}")
    file_path = os.path.join(ifp, fn)
    
    # Read the merged JSONL file
    with open(file_path, 'r') as f:
        data = json.load(f)

    random.shuffle(data)
    # Remove any elements with more than 20000 words
    filtered_data = []
    
    for entry in data:
        text = entry['context']
        word_count = len(text.split())
        answer_length = len(str(entry['answer']))
        if "No Answer Present" in str(entry['answer']) or "yes" == str(entry['answer']).lower() or "no" == str(entry['answer']).lower():
            continue
        if word_count <= 5000 and word_count > 10 and answer_length >= 3:
            filtered_data.append(entry)
    data = filtered_data


    data = data[:N]
    
    # Create output filename
    output_path = os.path.join(ofp, fn)
    
    # Write the top 1000 entries to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

print(f"Processed {len(fns)} files.")
