## Requirements
- Python 3.12
- Anaconda3
- CUDA 12.4 (for GPU support)

## Setup

### Step 1: Create and Activate Virtual Environment
Run the following commands to create and activate a virtual environment:

```shell
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

### Step 2: Install Dependencies
Run the following commands to install the required packages:

```shell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install jsonpickle transformers datasets matplotlib scikit-learn seqeval torchmetrics nltk accelerate evaluate peft bitsandbytes trl pynvml lm-eval levenshtein
uv pip install inspect-ai
```

## Usage



# Plan

Dataset of questions with detailed supporting context
    might need to construct this using google 
    for each question need to subset down to a set of chunks which are relevant? Test with and without this subset, as the model might do just fine when given the full context document
For each question, rewrite to include more specificity and detail, removing things which are implied or understood. Making those details explicit in the question.
Datasets
    start with mrqa_HotpotQA, mrqa_NaturalQuestionsShort, mrqa_TriviaQA-web
    ucinlp_drop might serve as a good test case of disambiguation. As many of the questions are not understandable in isolation.
    https://aclanthology.org/2024.emnlp-main.956.pdf
        https://github.com/Shaier/Adaptive_QA
