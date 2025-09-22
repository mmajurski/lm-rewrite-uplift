
source .venv/bin/activate











if [ "$1" == "pre" ]; then

model="gpt-oss-120b"
remote="pn131285:8443"

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

  
  src_fldr=./data-subset-500-afc/oe-gpt120b
  out_fldr=./data-subset-500-afc/oe-gpt120b-afc

  for dataset in "${datasets[@]}"; do
    python generate_answer_free_context.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
  done

  src_fldr=./data-subset-500-afc/oe-gpt120b-afc
  out_fldr=./data-subset-500-afc/oe-gpt120b-afc2

  for dataset in "${datasets[@]}"; do
    python generate_afc_reformat.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
  done

else

model="gpt-oss-120b"
remote="pn131285:8447"

datasets=(
  "ai_plan.json"
  "ai_plan_yb.json"
  "arXiv_2502_17521v1.json"
  "arXiv_2502_17521v1_yb.json"
  "hle.json"
)

  src_fldr=./data-post-cutoff-afc/oe-gpt120b
  out_fldr=./data-post-cutoff-afc/oe-gpt120b-afc

  # for dataset in "${datasets[@]}"; do
  #   python generate_answer_free_context.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
  # done


 src_fldr=./data-post-cutoff-afc/oe-gpt120b-afc
  out_fldr=./data-post-cutoff-afc/oe-gpt120b-afc2

  for dataset in "${datasets[@]}"; do
    python generate_afc_reformat.py --dataset=${dataset} --src_dataset_dir=${src_fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
  done


fi



# if [ "$1" == "gpt" ]; then
#   echo "Running GPT-OSS"
#   model="gpt-oss-120b"
#   remote="pn131285:8447"
#   out_fldr=./data-subset-500-SU2/oe-gpt120b

#   for dataset in "${datasets[@]}"; do
#     #python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
#     python self_uplift.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
#   done

# else
#   echo "Running Q235B"
#   model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
#   remote="pn131285:8446"
#   out_fldr=./data-subset-500-SU2/oe-Q235B

#   for dataset in "${datasets[@]}"; do
#     #python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
#     python self_uplift.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
#   done
# fi





# fldr=./data-post-cutoff/source_data/

# datasets=(
#   # "ai_plan.json"
#   # "ai_plan_yb.json"
#   # "arXiv_2502_17521v1.json"
#   # "arXiv_2502_17521v1_yb.json"
#   "hle.json"
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


