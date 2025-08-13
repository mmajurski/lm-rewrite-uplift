import os
import json
import random

ifp = './source_data'
N = 1000
ofp = f'./data-subset-1000'
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
        if word_count <= 10000:
            filtered_data.append(entry)
    data = filtered_data


    data = data[:N]
    
    # Create output filename
    output_path = os.path.join(ofp, fn)
    
    # Write the top 1000 entries to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

print(f"Processed {len(fns)} files.")
