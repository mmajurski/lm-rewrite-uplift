
source .venv/bin/activate

# datasets=(
# "ai_plan.json"
# "ai_plan_yb.json"
# "arXiv_2502_17521v1.json"
# "arXiv_2502_17521v1_yb.json"
# "hle.json"
# )

# model="gpt-oss-120b"
# remote="pn131285:8447"
# src_fldr=./data-post-cutoff/source_data/
# out_fldr=./data-post-cutoff-afc/source_data/
# for dataset in "${datasets[@]}"; do
#   python generate_answer_free_context.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
# done






datasets=(
"flashrag_2wikimultihopqa.json"
"flashrag_boolq.json"
"flashrag_fermi.json"
"flashrag_hotpotqa.json"
"flashrag_msmarcoqa.json"
"flashrag_musique.json"
"mrqa_NaturalQuestionsShort.json"
"mrqa_TriviaQA-web.json"
"squadv2.json"
"triva_qa.json"
)


# model="gpt-oss-120b"
# remote="pn131285:8447"

model="openai/gpt-oss-20b"
remote="iarpa018:8443"


src_fldr=./data-subset-500-afc/source_data
out_fldr=./data-subset-500-afc/oe-gpt20b-afc

for dataset in "${datasets[@]}"; do
  python generate_afc_reformat.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done




datasets=(
"ai_plan.json"
"ai_plan_yb.json"
"arXiv_2502_17521v1.json"
"arXiv_2502_17521v1_yb.json"
"hle.json"
)


model="openai/gpt-oss-20b"
remote="iarpa018:8443"

src_fldr=./data-post-cutoff-afc/source_data
out_fldr=./data-post-cutoff-afc/oe-gpt20b-afc

for dataset in "${datasets[@]}"; do
  python generate_afc_reformat.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done

model="gpt-oss-120b"
remote="pn131285:8447"

src_fldr=./data-post-cutoff-afc/source_data
out_fldr=./data-post-cutoff-afc/oe-gpt120b-afc

for dataset in "${datasets[@]}"; do
  python generate_afc_reformat.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done


model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
remote="pn131285:8446"

src_fldr=./data-post-cutoff-afc/source_data
out_fldr=./data-post-cutoff-afc/oe-Q235B-afc

for dataset in "${datasets[@]}"; do
  python generate_afc_reformat.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done










python evaluate_answer_giveaway.py
python evaluate_grounding.py
python evaluate_reformat_fidelity.py