
import os
import copy
from inspect_ai import eval
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model
# from inspect_ai.evals.arc import arc_challenge
# from inspect_ai.evals.arc import arc_challenge

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, solver, TaskState
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_FACT_TEMPLATE
import vllm_inspect_provider
import utils
import time
import json
import random

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# GRADER_MODEL = 'v_llm/Llama-4-Maverick-17B-128E-Instruct-FP8'
# GRADER_MODEL_BASE_URL = 'https://rchat.nist.gov/api'
# GRADER_MODEL_API_KEY=os.getenv("RCHAT_API_KEY")

# GRADER_MODEL="v_llm/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# GRADER_MODEL_BASE_URL="https://pn131285.nist.gov:8443/v1"
# GRADER_MODEL_API_KEY=os.getenv("VLLM_API_KEY")

GRADER_MODEL="v_llm/gpt-oss-120b"
GRADER_MODEL_BASE_URL="https://pn131285.nist.gov:8447/v1"
GRADER_MODEL_API_KEY=os.getenv("VLLM_API_KEY")

# GRADER_MODEL="v_llm/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
# GRADER_MODEL_BASE_URL="https://pn131285.nist.gov:8446/v1"
# GRADER_MODEL_API_KEY=os.getenv("VLLM_API_KEY")






@solver
def open_ended_qa() -> Solver:

    template = r"""
Answer the following open ended short answer question. The last line of your response should be of the following format: 'ANSWER: $answer' (without quotes) where answer is the answer to the question. Think step by step before answering.

{question}
""".strip()
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:

        state.user_prompt.text = template.format(question=state.user_prompt.text)
        state = await generate(state)

        val = state.output.completion
        val = val.strip()
        val = val.split('\n')
        if val[-1].startswith('ANSWER: '):
            val = val[-1]
            state.output.completion = val

        return state

    return solve




def base_task(lcl_fp, question_key):

    config = GenerateConfig(max_connections=64, timeout=300)
    model = get_model(model=GRADER_MODEL, base_url=GRADER_MODEL_BASE_URL, config=config, api_key=GRADER_MODEL_API_KEY)

    
    with open(lcl_fp, 'r') as f:
        ds = json.load(f)
    
    # random.shuffle(ds)
    samples = list()
    for row in ds:
        context = row['context']
        question = row[question_key]
        input = f"<context>{context}</context>\n\n\n\nQuestion: {question}"

        samples.append(Sample(input=input, target=str(row['orig_answer'])))
    
    return Task(
        dataset = samples,
        solver=open_ended_qa(),
        scorer=model_graded_qa(model=model, template=DEFAULT_MODEL_GRADED_FACT_TEMPLATE),
        fail_on_error=False,
    )





@task
def flashrag_2wikimultihopqa(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)
    
@task
def flashrag_boolq(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def flashrag_fermi(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def flashrag_hotpotqa(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def flashrag_msmarcoqa(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def flashrag_musique(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def mrqa_HotpotQA(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def mrqa_NaturalQuestionsShort(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def mrqa_TriviaQA_web(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def natural_questions(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def squadv2(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def triva_qa(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)


@task
def ai_plan(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def ai_plan_yb(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def arXiv_2502_17521v1(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def hle(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def arXiv_2502_17521v1_yb(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)


def get_task_dir_dict(dataset_fldr):
    return {
        'flashrag_2wikimultihopqa': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_2wikimultihopqa.json')),
        'flashrag_boolq': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_boolq.json')),
        'flashrag_fermi': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_fermi.json')),
        'flashrag_hotpotqa': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_hotpotqa.json')),
        'flashrag_msmarcoqa': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_msmarcoqa.json')),
        'flashrag_musique': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_musique.json')),
        'mrqa_HotpotQA': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_HotpotQA.json')),
        'mrqa_NaturalQuestionsShort': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_NaturalQuestionsShort.json')),
        'mrqa_TriviaQA_web': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_TriviaQA-web.json')),
        'squadv2': os.path.abspath(os.path.join(dataset_fldr, 'squadv2.json')),
        'triva_qa': os.path.abspath(os.path.join(dataset_fldr, 'triva_qa.json')),
        'hle': os.path.abspath(os.path.join(dataset_fldr, 'hle.json')),
        'ai_plan': os.path.abspath(os.path.join(dataset_fldr, 'ai_plan.json')),
        'ai_plan_yb': os.path.abspath(os.path.join(dataset_fldr, 'ai_plan_yb.json')),
        'arXiv_2502_17521v1': os.path.abspath(os.path.join(dataset_fldr, 'arXiv_2502_17521v1.json')),
        'arXiv_2502_17521v1_yb': os.path.abspath(os.path.join(dataset_fldr, 'arXiv_2502_17521v1_yb.json')),
    }

def get_task(name: str, dataset_fldr: str, question_key: str):
    task_map = {
        'flashrag_2wikimultihopqa': flashrag_2wikimultihopqa,
        'flashrag_boolq': flashrag_boolq,
        'flashrag_fermi': flashrag_fermi,
        'flashrag_hotpotqa': flashrag_hotpotqa,
        'flashrag_msmarcoqa': flashrag_msmarcoqa,
        'flashrag_musique': flashrag_musique,
        'mrqa_HotpotQA': mrqa_HotpotQA,
        'mrqa_NaturalQuestionsShort': mrqa_NaturalQuestionsShort,
        'mrqa_TriviaQA_web': mrqa_TriviaQA_web,
        'squadv2': squadv2,
        'triva_qa': triva_qa,
        'hle': hle,
        'ai_plan': ai_plan,
        'ai_plan_yb': ai_plan_yb,
        'arXiv_2502_17521v1': arXiv_2502_17521v1,
        'arXiv_2502_17521v1_yb': arXiv_2502_17521v1_yb,
    }
    try:
        return task_map[name](dataset_fldr, question_key)
    except KeyError:
        raise ValueError(f"Task {name} not found")




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluates a MMLU style dataset using Inspect framework.')
    # parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    # parser.add_argument('--test_path', type=str, default='./data/squad_reformat_open', help='Path to the test dataset, specifically the folder containing the json dataset files.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--question_key', type=str, default='orig_question')#, required=True)  # ['orig_question', 'question']
    parser.add_argument('--base_dir', type=str, required=True)

    args = parser.parse_args()
    question_key = args.question_key
    if question_key not in ['orig_question', 'question']:
        raise ValueError(f"Invalid question key: {question_key}")


    config = GenerateConfig(max_connections=args.batch_size, timeout=300) # max_tokens=8192
    models = list()


    models_dict = dict()

    
    # models_dict['gpt-oss-120b'] = get_model(model="v_llm/gpt-oss-120b", base_url="https://pn131285.nist.gov:8447/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['openai/gpt-oss-20b'] = get_model(model="v_llm/openai/gpt-oss-20b", base_url="https://iarpa018.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    # models_dict['google/gemma-3-270m-it'] = get_model(model="v_llm/google/gemma-3-270m-it", base_url="https://pn120393.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['google/gemma-3-1b-it'] = get_model(model="v_llm/google/gemma-3-1b-it", base_url="https://pn120393.nist.gov:8444/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['google/gemma-3-4b-it'] = get_model(model="v_llm/google/gemma-3-4b-it", base_url="https://pn120393.nist.gov:8445/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['google/gemma-3-12b-it'] = get_model(model="v_llm/google/gemma-3-12b-it", base_url="https://pn120393.nist.gov:8446/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['google/gemma-3-27b-it'] = get_model(model="v_llm/google/gemma-3-27b-it", base_url="https://pn125915.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    # models_dict['meta-llama/Llama-3.2-3B-Instruct'] = get_model(model="v_llm/meta-llama/Llama-3.2-3B-Instruct", base_url="https://pn125915.nist.gov:8444/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['meta-llama/Llama-3.1-8B-Instruct'] = get_model(model="v_llm/meta-llama/Llama-3.1-8B-Instruct", base_url="https://pn125915.nist.gov:8445/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
        
    # models_dict['microsoft/phi-4'] = get_model(model="v_llm/microsoft/phi-4", base_url="https://pn125916.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    # models_dict['Qwen/Qwen3-0.6B'] = get_model(model="v_llm/Qwen/Qwen3-0.6B", base_url="https://pn125916.nist.gov:8444/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['Qwen/Qwen3-1.7B'] = get_model(model="v_llm/Qwen/Qwen3-1.7B", base_url="https://pn125916.nist.gov:8445/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['Qwen/Qwen3-4B-Instruct-2507'] = get_model(model="v_llm/Qwen/Qwen3-4B-Instruct-2507", base_url="https://pn125916.nist.gov:8446/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['Qwen/Qwen2.5-7B-Instruct'] = get_model(model="v_llm/Qwen/Qwen2.5-7B-Instruct", base_url="https://pn125917.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['Qwen/Qwen3-30B-A3B-Instruct-2507'] = get_model(model="v_llm/Qwen/Qwen3-30B-A3B-Instruct-2507", base_url="https://iarpa017.nist.gov:8444/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    # models_dict['Qwen/Qwen3-235B-A22B-Instruct-2507-FP8'] = get_model(model="v_llm/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", base_url="https://pn131285.nist.gov:8446/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    


    models_dict['meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'] = get_model(model="v_llm/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", base_url="https://pn131275.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))
    models_dict['meta-llama/Llama-3.3-70B-Instruct'] = get_model(model="v_llm/meta-llama/Llama-3.3-70B-Instruct", base_url="https://pn131285.nist.gov:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    # models_dict['openai/gpt-5'] = get_model(model="openai/gpt-5", base_url="https://api.openai.com/v1", config=config, api_key=os.getenv("OPENAI_API_KEY"))
    # models_dict['openai/gpt-5-mini'] = get_model(model="openai/gpt-5-mini", base_url="https://api.openai.com/v1", config=config, api_key=os.getenv("OPENAI_API_KEY"))
    # models_dict['openai/gpt-5-nano'] = get_model(model="openai/gpt-5-nano", base_url="https://api.openai.com/v1", config=config, api_key=os.getenv("OPENAI_API_KEY"))



    

    # base_dir = './data-subset-500'
    # base_dir = './data-post-cutoff'
    base_dir = args.base_dir
    # ds = 'oe-Q235B'

    # # # disp_type = 'plain'  # full, conversation, rich, plain, none
    disp_type = 'full'



    available_models = list(models_dict.keys())

    for ds in ['oe-Q235B-filtered', 'oe-gpt120b-filtered']:
        dataset_fldr = f"{base_dir}/{ds}"
        if not os.path.exists(dataset_fldr):
            continue

        print("--------------------------------")
        print(f"Processing folder {ds}")

        

        available_task_names_dict = get_task_dir_dict(dataset_fldr)
        # first_key = next(iter(available_task_names_dict))
        # available_task_names_dict = {first_key: available_task_names_dict[first_key]}


        to_remove = list()
        for k in available_task_names_dict.keys():
            if not os.path.exists(available_task_names_dict[k]):
                to_remove.append(k)
                print("missing task: ", k, " at ", available_task_names_dict[k])
        for k in to_remove:
            available_task_names_dict.pop(k)
        available_task_names = list(available_task_names_dict.keys())



        
        if question_key == 'orig_question':
            log_dir = os.path.join(base_dir, f"logs-{ds}-orig-giveaway")  
        else:
            log_dir = os.path.join(base_dir, f"logs-{ds}-reformat-giveaway") 

        print("discovering completed logs...")
        completed_logs, completed_fns = utils.get_completed_logs(log_dir)
        for l_idx, log in enumerate(completed_logs):
            d_fp = log['task_args']['dataset_fldr']
            if ds not in d_fp:
                print(f"Log in Wrong Place:  {completed_fns[l_idx]}")


        unused_logs = copy.deepcopy(completed_logs)

        
        
        for task_name in available_task_names:
            print("--------------------------------")
            print(f"Processing folder {dataset_fldr} task {task_name}")
            work_models = list(set(available_models))

            to_remove_models = []
            for log in completed_logs:
                if log['task'] == task_name:
                    if log['model'].replace('v_llm/', '') in available_models:
                        to_remove_models.append(log['model'].replace('v_llm/', ''))
                        unused_logs.remove(log)
            work_models = list(set(available_models) - set(to_remove_models))

            if len(work_models) == 0:
                print(f"  No missing models")
                continue

            
            print(f"  Missing {len(work_models)} Models:")
            for model in work_models:
                print(f"    {model}")
            print("--------------------------------")
            print("starting in 3...")
            time.sleep(1)
            print("            2...")
            time.sleep(1)
            print("            1...")
            time.sleep(1)

            models_list = [models_dict[model] for model in work_models]
            json_fp = available_task_names_dict[task_name]
            work_tasks = [get_task(task_name, json_fp, question_key)]
            eval(work_tasks, model=models_list, display=disp_type, log_format='json', no_log_images=True, no_log_samples=True, log_dir=log_dir, max_connections=64) #, max_subprocesses=64) #, timeout=300)
            