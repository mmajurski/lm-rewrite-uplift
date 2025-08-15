import os
import json
import numpy as np


ifp1 = './data-subset-500/oe-gpt120b/'
ifp2 = './data-subset-500/oe-Q235B/'

ofp1 = './data-subset-500/oe-gpt120b-filtered/'
ofp2 = './data-subset-500/oe-Q235B-filtered/'
os.makedirs(ofp1, exist_ok=True)
os.makedirs(ofp2, exist_ok=True)

fns = [f for f in os.listdir(ifp1) if f.endswith('.json')]
fns.sort()
# st_idx = 4
# fns = fns[st_idx:st_idx+1]

score_thres = 5


for fn in fns:
    print(80*"=")
    print(f"fn: {fn}")
    print(80*"=")
    rewrite_gpt_scores = []
    rewrite_qwen_scores = []

    with open(os.path.join(ifp1, fn), 'r') as f:
        data_gpt = json.load(f)
    with open(os.path.join(ifp2, fn), 'r') as f:
        data_qwen = json.load(f)
    
    print(f"len(data_gpt): {len(data_gpt)}")
    print(f"len(data_qwen): {len(data_qwen)}")

    # Filter to keep only questions that exist in both datasets based on orig_question
    gpt_orig_questions = {d.get('orig_question') for d in data_gpt if d.get('orig_question') is not None}
    qwen_orig_questions = {d.get('orig_question') for d in data_qwen if d.get('orig_question') is not None}
    
    # Find common orig_questions
    common_questions = gpt_orig_questions & qwen_orig_questions
    print(f"Common orig_questions: {len(common_questions)}")
    
    # Filter both datasets to only include questions with common orig_question
    data_gpt = [d for d in data_gpt if d.get('orig_question') in common_questions]
    data_qwen = [d for d in data_qwen if d.get('orig_question') in common_questions]
    
    print(f"After orig_question filtering - len(data_gpt): {len(data_gpt)}")
    print(f"After orig_question filtering - len(data_qwen): {len(data_qwen)}")
    
    # Poor man's histogram for reformat_answer_giveaway_score in data_gpt
    print("Histogram for GPT reformat_answer_giveaway_score:")
    gpt_hist = {}
    for d in data_gpt:
        score = d.get('reformat_answer_giveaway_score')
        if score is not None:
            gpt_hist[score] = gpt_hist.get(score, 0) + 1
    for score in sorted(gpt_hist):
        print(f"Score {score}: {gpt_hist[score]}")

    # Poor man's histogram for reformat_answer_giveaway_score in data_qwen
    print("Histogram for Qwen reformat_answer_giveaway_score:")
    qwen_hist = {}
    for d in data_qwen:
        score = d.get('reformat_answer_giveaway_score')
        if score is not None:
            qwen_hist[score] = qwen_hist.get(score, 0) + 1
    for score in sorted(qwen_hist):
        print(f"Score {score}: {qwen_hist[score]}")
    print()


    for d_gpt, d_qwen in zip(data_gpt, data_qwen):
        if 'reformat_answer_giveaway_score' in d_gpt and 'reformat_answer_giveaway_score' in d_qwen:    
            rewrite_gpt_scores.append(d_gpt['reformat_answer_giveaway_score'])
            rewrite_qwen_scores.append(d_qwen['reformat_answer_giveaway_score'])
        else:
            rewrite_gpt_scores.append(None)
            rewrite_qwen_scores.append(None)


    mask1 = np.asarray(rewrite_gpt_scores) <= score_thres
    mask2 = np.asarray(rewrite_qwen_scores) <= score_thres
    
    print(f"gpt scores count: {np.sum(mask1)}")
    print(f"qwen scores count: {np.sum(mask2)}")
    mask = mask1 & mask2

    data_gpt = [d_gpt for i, d_gpt in enumerate(data_gpt) if mask[i]]
    data_qwen = [d_qwen for i, d_qwen in enumerate(data_qwen) if mask[i]]
    gpt_scores = [d_gpt['reformat_answer_giveaway_score'] for d_gpt in data_gpt]
    qwen_scores = [d_qwen['reformat_answer_giveaway_score'] for d_qwen in data_qwen]
    print(f"filtered gpt scores mean: {np.mean(gpt_scores)}")
    print(f"filtered gpt scores max: {np.max(gpt_scores)}")
    print(f"filtered gpt scores min: {np.min(gpt_scores)}")
    print(f"filtered qwen scores mean: {np.mean(qwen_scores)}")
    print(f"filtered qwen scores max: {np.max(qwen_scores)}")
    print(f"filtered qwen scores min: {np.min(qwen_scores)}")
    
    print(f"Saving {len(data_gpt)} questions to {ofp1}")
    with open(os.path.join(ofp1, fn), 'w') as f:
        json.dump(data_gpt, f, indent=2)
    print(f"Saving {len(data_qwen)} questions to {ofp2}")
    with open(os.path.join(ofp2, fn), 'w') as f:
        json.dump(data_qwen, f, indent=2)







        








