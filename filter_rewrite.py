import os
import json
import numpy as np



score_thres = 5

# ifp = './data-subset-500'
ifp = './data-post-cutoff'
fldrs = ['oe-Q235B', 'oe-gpt120b']

for fldr in fldrs:
    cur_ifp = os.path.join(ifp, fldr)
    cur_ofp = os.path.join(ifp, fldr + '-filtered')
    os.makedirs(cur_ofp, exist_ok=True)

    fns = [f for f in os.listdir(cur_ifp) if f.endswith('.json')]
    fns.sort()

    for fn in fns:
        print(80*"=")
        print(f"fn: {fn}")
        print(80*"=")

        
        with open(os.path.join(cur_ifp, fn), 'r') as f:
            data = json.load(f)

        orig_questions = {d.get('orig_question') for d in data if d.get('orig_question') is not None}

        # Poor man's histogram for reformat_answer_giveaway_score in data_gpt
        # print("Histogram for GPT reformat_answer_giveaway_score:")
        # gpt_hist = {}
        # for d in data:
        #     score = d.get('reformat_answer_giveaway_score')
        #     if score is not None:
        #         gpt_hist[score] = gpt_hist.get(score, 0) + 1
        # for score in sorted(gpt_hist):
        #     print(f"Score {score}: {gpt_hist[score]}")

        giveaway_scores = np.asarray([d['reformat_answer_giveaway_score'] for d in data])
        giveaway_orig_scores = np.asarray([d['orig_answer_giveaway_score'] for d in data])
        answer_sim_scores = np.asarray([d['reformat_answer_similarity_score'] for d in data])
        question_sim_scores = np.asarray([d['reformat_question_similarity_score'] for d in data])
        mask1 = giveaway_scores <= giveaway_orig_scores + 1
        mask1[giveaway_scores < score_thres] = True
        mask2 = answer_sim_scores >= score_thres
        mask3 = question_sim_scores >= score_thres
        # mask4 = giveaway_orig_scores < score_thres
        mask = mask1 & mask2 & mask3# & mask4

        scores = [d['reformat_answer_giveaway_score'] for d in data]
        print(f"filtered reformat_answer_giveaway_score mean: {np.mean(scores)}")
        print(f"filtered reformat_answer_giveaway_score max: {np.max(scores)}")
        print(f"filtered reformat_answer_giveaway_score min: {np.min(scores)}")
        print()

        # scores = [d['orig_answer_giveaway_score'] for d in data]
        # print(f"filtered orig_answer_giveaway_score mean: {np.mean(scores)}")
        # print(f"filtered orig_answer_giveaway_score max: {np.max(scores)}")
        # print(f"filtered orig_answer_giveaway_score min: {np.min(scores)}")
        # print()

        scores = [d['reformat_answer_similarity_score'] for d in data]
        print(f"filtered reformat_answer_similarity_score mean: {np.mean(scores)}")
        print(f"filtered reformat_answer_similarity_score max: {np.max(scores)}")
        print(f"filtered reformat_answer_similarity_score min: {np.min(scores)}")
        print()

        scores = [d['reformat_question_similarity_score'] for d in data]
        print(f"filtered reformat_question_similarity_score mean: {np.mean(scores)}")
        print(f"filtered reformat_question_similarity_score max: {np.max(scores)}")
        print(f"filtered reformat_question_similarity_score min: {np.min(scores)}")
        print()

        data = [d_gpt for i, d_gpt in enumerate(data) if mask[i]]

        

        print(f"Saving {len(data)} questions to {cur_ofp}")

        # if 'hle' in fn:
        #     for d_idx in range(len(data)):
        #         if not mask[d_idx]:
        #             print(data[d_idx])
        with open(os.path.join(cur_ofp, fn), 'w') as f:
            json.dump(data, f, indent=2)
        








        








