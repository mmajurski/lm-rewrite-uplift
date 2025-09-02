
source .venv/bin/activate






fldr=./data-subset-500-SU/source_data/

datasets=(
  "flashrag_2wikimultihopqa.json"
  "flashrag_boolq.json"
  "flashrag_fermi.json"
  "flashrag_hotpotqa.json"
  "flashrag_msmarcoqa.json"
  "flashrag_musique.json"
  "mrqa_HotpotQA.json"
  "mrqa_NaturalQuestionsShort.json"
  "mrqa_TriviaQA-web.json"
  "squadv2.json"
  "triva_qa.json"
)



if [ "$1" == "gpt" ]; then
  echo "Running GPT-OSS"
  model="gpt-oss-120b"
  remote="pn131285:8447"
  out_fldr=./data-subset-500-SU/oe-gpt120b

  for dataset in "${datasets[@]}"; do
    #python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
    python self_uplift.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
  done

else
  echo "Running Q235B"
  model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
  remote="pn131285:8446"
  out_fldr=./data-subset-500-SU/oe-Q235B

  for dataset in "${datasets[@]}"; do
    #python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
    python self_uplift.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
  done
fi





# fldr=./data-post-cutoff/source_data/

# datasets=(
#   "ai_plan.json"
#   "ai_plan_yb.json"
#   "arXiv_2502_17521v1.json"
#   "hle.json"
#   "arXiv_2502_17521v1_yb.json"
# )




# model="gpt-oss-120b"
# remote="pn131285:8447"
# out_fldr=./data-post-cutoff/oe-gpt120b


# for dataset in "${datasets[@]}"; do
#   python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
# done


# model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
# remote="pn131285:8446"
# out_fldr=./data-post-cutoff/oe-Q235B


# for dataset in "${datasets[@]}"; do
#   python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
# done


