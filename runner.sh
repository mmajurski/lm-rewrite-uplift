
source .venv/bin/activate


if [ "$1" == "post" ]; then
echo "Running Post Cutoff Data"

python inspect_eval_open.py --question_key='orig_question' --base_dir='./data-post-cutoff'
python inspect_eval_open.py --question_key='question' --base_dir='./data-post-cutoff'

python inspect_eval_open_giveaway.py --question_key='orig_question' --base_dir='./data-post-cutoff'
python inspect_eval_open_giveaway.py --question_key='question' --base_dir='./data-post-cutoff'


else

echo "Running Baseline Data"

python inspect_eval_open.py --question_key='orig_question' --base_dir='./data-subset-500'
python inspect_eval_open.py --question_key='question' --base_dir='./data-subset-500'

python inspect_eval_open_giveaway.py --question_key='orig_question' --base_dir='./data-subset-500'
python inspect_eval_open_giveaway.py --question_key='question' --base_dir='./data-subset-500'

fi





