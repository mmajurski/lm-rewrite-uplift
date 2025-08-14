
source .venv/bin/activate






fldr=./data-subset-500/source_data/

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
  "pubmed_qa.json"
  "squadv2.json"
  "triva_qa.json"
)

# if [ "$1" == "gpt" ]; then
#     echo "Running with gpt-oss-120b"
#     model="gpt-oss-120b"
#     remote="pn131285:8447"
#     out_fldr=./data-subset-200/oe-gpt120b
# elif [ "$1" == "qwen" ]; then
#     echo "Running with Qwen3-235B-A22B-Instruct-2507-FP8"
#     model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
#     remote="pn131285:8446"
#     out_fldr=./data-subset-200/oe-Q235B
# else
#     echo "Invalid model: $1"
#     exit 1
# fi

# for dataset in "${datasets[@]}"; do
#   python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
# done


model="gpt-oss-120b"
remote="pn131285:8447"
out_fldr=./data-subset-500/oe-gpt120b


for dataset in "${datasets[@]}"; do
  python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done


model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
remote="pn131285:8446"
out_fldr=./data-subset-500/oe-Q235B


for dataset in "${datasets[@]}"; do
  python generate_reformat.py --dataset=${dataset} --src_dataset_dir=${fldr} --out_dataset_dir=${out_fldr} --remote=${remote} --model=${model}
done


