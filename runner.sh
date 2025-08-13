
source .venv/bin/activate





python inspect_eval_open.py --question_key='orig_question'
python inspect_eval_open.py --question_key='question'


# fldr=./data-subset-1000/oe-unmodified/
# # model=Llama-4-Maverick-17B-128E-Instruct-FP8
# # # remote=rchat
# # remote="pn131285:8443"

# model="gpt-oss-120b"
# remote="pn131285:8447"
# out_fldr=./data-subset-1000/oe-gpt120b


# datasets=(
#   "flashrag_2wikimultihopqa.json"
#   "flashrag_boolq.json"
#   "flashrag_fermi.json"
#   "flashrag_hotpotqa.json"
#   "flashrag_msmarcoqa.json"
#   "flashrag_musique.json"
#   "mrqa_HotpotQA.json"
#   "mrqa_NaturalQuestionsShort.json"
#   "mrqa_TriviaQA-web.json"
#   "pubmed_qa.json"
#   "squadv2.json"
#   "triva_qa.json"
# )




# # for dataset in "${datasets[@]}"; do
# #   python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
# # done




# model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
# remote="pn131285:8446"
# out_fldr=./data-subset-1000/oe-Q235B


# for dataset in "${datasets[@]}"; do
#   python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
# done
