
import os
import copy
import utils
import time
import json
import numpy as np
import pandas as pd






output_dict = list()

for base_dir in ['./data-subset-500-emb', './data-post-cutoff-emb']:
    for ds in ['oe-Q235B-filtered', 'oe-gpt120b-filtered']:
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
            cosine_embO_embOwC = dict()
            cosine_embR_embOwC = dict()
            for mn in model_names:
                cosine_embO_embOwC[mn] = list()
                cosine_embR_embOwC[mn] = list()

            for sample in data:
                for mn in model_names:
                    cosine_embO_embOwC[mn].append(sample[f'{mn}_embeddings']['cosine_embO_embOwC'])
                    cosine_embR_embOwC[mn].append(sample[f'{mn}_embeddings']['cosine_embR_embOwC'])
            
            
            for mn in model_names:
                a = np.mean(cosine_embO_embOwC[mn])
                b = np.mean(cosine_embR_embOwC[mn])
                row = {'dataset': ds, 'benchmark': json_fn.replace('.json',''), 'embedding_model': mn, 'cosine_embO_embOwC':a, 'cosine_embR_embOwC': b}
                output_dict.append(row)


df = pd.DataFrame(output_dict)
df.to_csv('post_cutoff_q_embeddings.csv', index=False)



    
    