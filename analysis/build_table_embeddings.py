"""Build summary tables of embedding similarity metrics across datasets and models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import os
import copy
import utils
import time
import json
import numpy as np
import pandas as pd






output_dict = list()

for base_dir in ['./data-subset-500-emb', './data-post-cutoff-emb']:
    for ds in ['oe-Q235B-filtered', 'oe-gpt120b-filtered', 'oe-gpt120b-filtered-afc']:
        dataset_fldr = os.path.join(base_dir, ds)
        if not os.path.exists(dataset_fldr):
            continue

        json_files = [fn for fn in os.listdir(dataset_fldr) if fn.endswith('.json')]
        for json_fn in json_files:
            cur_d_fp = os.path.join(dataset_fldr, json_fn)
            with open(cur_d_fp, 'r') as f:
                data = json.load(f)

            print("--------------------------------")
            print(f"Processing {cur_d_fp} embeddings")

            sample = data[0]
            keys = [k for k in sample.keys() if k.endswith('_embeddings')]
            model_names = [m.replace('_embeddings', '') for m in keys]
            cosine_embO_embC = dict()
            cosine_embR_embC = dict()
            for mn in model_names:
                cosine_embO_embC[mn] = list()
                cosine_embR_embC[mn] = list()

            for sample in data:
                for mn in model_names:
                    cosine_embO_embC[mn].append(sample[f'{mn}_embeddings']['cosine_embO_embC'])
                    cosine_embR_embC[mn].append(sample[f'{mn}_embeddings']['cosine_embR_embC'])
            
            
            for mn in model_names:
                a = np.mean(cosine_embO_embC[mn])
                b = np.mean(cosine_embR_embC[mn])
                row = {'dataset': ds, 'benchmark': json_fn.replace('.json',''), 'embedding_model': mn, 'cosine_embO_embC':a, 'cosine_embR_embC': b}
                output_dict.append(row)


# Group by embedding_model and create 3 CSV files instead of 1

df = pd.DataFrame(output_dict)

df['cosine_diff_embR_minus_embO'] = df['cosine_embR_embC'] - df['cosine_embO_embC']


# 1. Save the full table as before
df.to_csv('./understanding/post_cutoff_q_embeddings.csv', index=False)

# 2. Group by embedding_model and save one CSV per model
for embedding_model, group_df in df.groupby('embedding_model'):
    group_df.to_csv(f'./understanding/q_embeddings_{embedding_model}.csv', index=False)

# 3. Save a summary CSV with mean values per embedding_model
summary_df = df.groupby('embedding_model')[['cosine_embO_embC', 'cosine_embR_embC', 'cosine_diff_embR_minus_embO']].mean().reset_index()
summary_df.to_csv('./understanding/q_embeddings_summary.csv', index=False)




    
    