
source ~/venv/lmg/bin/activate


# python inspect_eval.py --run_reformat
# python inspect_eval.py

# python inspect_eval_open.py --run_reformat
# python inspect_eval_open.py






fldr=./data-subset-100
# model=Llama-4-Maverick-17B-128E-Instruct-FP8
# # remote=rchat
# remote="pn131285:8443"

model="gpt-4.1-mini"
remote="openai"


datasets=(
  # "mrqa_NaturalQuestionsShort.jsonl"
  # "ucinlp_drop.jsonl"

  # "annurev-control-071020-104336.jsonl"
  # "arXiv_2502_17521v1.jsonl"
  "squadv2.jsonl"
  "mrqa_HotpotQA.jsonl"
  "mrqa_TriviaQA-web.jsonl"
  "pubmed_qa.jsonl"
  "sec_qa.jsonl"
)



out_fldr=./data-subset-100/open-gpt_4_1_mini
for dataset in "${datasets[@]}"; do
  python generate_novel_open.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done