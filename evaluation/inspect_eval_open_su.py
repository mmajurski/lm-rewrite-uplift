"""Inspect-ai evaluation runner: self-uplift (SU) rewritten questions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import os
import copy
from inspect_ai import eval
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, solver, TaskState
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_FACT_TEMPLATE
import vllm_inspect_provider
import utils
import time
import json

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Grader model configuration — set these in your .env file
GRADER_MODEL = os.getenv("GRADER_MODEL", "v_llm/gpt-oss-120b")
GRADER_MODEL_BASE_URL = os.getenv("GRADER_MODEL_BASE_URL")
GRADER_MODEL_API_KEY = os.getenv("GRADER_MODEL_API_KEY") or os.getenv("VLLM_API_KEY")

if GRADER_MODEL_BASE_URL is None:
    raise ValueError("GRADER_MODEL_BASE_URL environment variable is not set. See .env.example.")


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

    config = GenerateConfig(max_connections=256, timeout=300)
    model = get_model(model=GRADER_MODEL, base_url=GRADER_MODEL_BASE_URL, config=config, api_key=GRADER_MODEL_API_KEY)

    with open(lcl_fp, 'r') as f:
        ds = json.load(f)

    samples = list()
    for row in ds:
        samples.append(Sample(input=row[question_key], target=str(row['orig_answer'])))

    return Task(
        dataset=samples,
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
def arXiv_2502_17521v1_yb(dataset_fldr, question_key):
    return base_task(dataset_fldr, question_key)

@task
def hle(dataset_fldr, question_key):
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


def load_models_config(config):
    """Load evaluation model configurations from models_config.json.

    See models_config.json.example for the expected format.
    """
    models_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models_config.json')
    if not os.path.exists(models_config_path):
        raise FileNotFoundError(
            f"models_config.json not found at {models_config_path}. "
            "Copy models_config.json.example to models_config.json and configure your server endpoints."
        )
    with open(models_config_path, 'r') as f:
        models_config = json.load(f)

    models_dict = dict()
    for model_name, cfg in models_config.items():
        api_key = os.getenv(cfg.get('api_key_env', 'VLLM_API_KEY'), 'sk-no-key-required')
        models_dict[model_name] = get_model(
            model=cfg['model_id'],
            base_url=cfg['base_url'],
            config=config,
            api_key=api_key
        )
    return models_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluates a dataset using the Inspect framework.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--question_key', type=str, default='question')  # ['orig_question', 'question']

    args = parser.parse_args()
    question_key = args.question_key
    if question_key not in ['orig_question', 'question']:
        raise ValueError(f"Invalid question key: {question_key}")

    config = GenerateConfig(max_connections=args.batch_size, timeout=300)
    models_dict = load_models_config(config)

    base_dir = './data-subset-500-SU2'
    disp_type = 'full'

    available_models = list(models_dict.keys())

    for ds in ['oe-Q235B-filtered', 'oe-gpt120b-filtered']:
        dataset_fldr = f"{base_dir}/{ds}"
        if not os.path.exists(dataset_fldr):
            continue
        print("--------------------------------")
        print(f"Processing folder {ds}")

        available_task_names_dict = get_task_dir_dict(dataset_fldr)

        to_remove = list()
        for k in available_task_names_dict.keys():
            if not os.path.exists(available_task_names_dict[k]):
                to_remove.append(k)
                print("missing task: ", k, " at ", available_task_names_dict[k])
        for k in to_remove:
            available_task_names_dict.pop(k)
        available_task_names = list(available_task_names_dict.keys())

        if question_key == 'orig_question':
            log_dir = os.path.join(base_dir, f"logs-{ds}-orig")
        else:
            log_dir = os.path.join(base_dir, f"logs-{ds}-reformat")

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
            eval(work_tasks, model=models_list, display=disp_type, log_format='json',
                 no_log_images=True, no_log_samples=True, log_dir=log_dir, max_connections=64)
