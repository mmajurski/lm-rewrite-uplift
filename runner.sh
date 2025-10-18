
source .venv/bin/activate







# python inspect_eval_open.py --question_key='orig_question' --base_dir='./data-subset-500'
# python inspect_eval_open.py --question_key='reformat_question' --base_dir='./data-subset-500'

# echo "*************************"
# echo "*************************"
# echo "BLOCK 1 Complete"
# echo "*************************"
# echo "*************************"

python inspect_eval_open_afc.py --question_key='orig_question' --base_dir='./data-subset-500-afc'
python inspect_eval_open_afc.py --question_key='reformat_question' --base_dir='./data-subset-500-afc'

echo "*************************"
echo "*************************"
echo "BLOCK 2 Complete"
echo "*************************"
echo "*************************"

# python inspect_eval_open.py --question_key='orig_question' --base_dir='./data-post-cutoff'
# python inspect_eval_open.py --question_key='reformat_question' --base_dir='./data-post-cutoff'

# echo "*************************"
# echo "*************************"
# echo "BLOCK 3 Complete"
# echo "*************************"
# echo "*************************"

python inspect_eval_open_afc.py --question_key='orig_question' --base_dir='./data-post-cutoff-afc'
python inspect_eval_open_afc.py --question_key='reformat_question' --base_dir='./data-post-cutoff-afc'

echo "*************************"
echo "*************************"
echo "BLOCK 4 Complete"
echo "*************************"
echo "*************************"

# python inspect_eval_open_giveaway.py --question_key='orig_question' --base_dir='./data-subset-500'
# python inspect_eval_open_giveaway.py --question_key='reformat_question' --base_dir='./data-subset-500'


# echo "*************************"
# echo "*************************"
# echo "BLOCK 5 Complete"
# echo "*************************"
# echo "*************************"


# python inspect_eval_open_giveaway.py --question_key='orig_question' --base_dir='./data-post-cutoff'
# python inspect_eval_open_giveaway.py --question_key='reformat_question' --base_dir='./data-post-cutoff'

# echo "*************************"
# echo "*************************"
# echo "BLOCK 6 Complete"
# echo "*************************"
# echo "*************************"


python inspect_eval_open_giveaway_afc.py --question_key='orig_question' --base_dir='./data-subset-500-afc'
python inspect_eval_open_giveaway_afc.py --question_key='reformat_question' --base_dir='./data-subset-500-afc'

echo "*************************"
echo "*************************"
echo "BLOCK 7 Complete"
echo "*************************"
echo "*************************"

python inspect_eval_open_giveaway_afc.py --question_key='orig_question' --base_dir='./data-post-cutoff-afc'
python inspect_eval_open_giveaway_afc.py --question_key='reformat_question' --base_dir='./data-post-cutoff-afc'

