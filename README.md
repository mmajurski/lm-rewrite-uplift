# Query Disambiguation via Answer-Free Context

This repository contains the code and experiments for the paper:

> **Query Disambiguation via Answer-Free Context**
> *[Authors, Venue, Year]*

We investigate whether making benchmark questions more explicit — by attaching a brief, answer-free grounding context — improves open-ended QA accuracy across a range of LLMs and QA datasets.

---

## Repository Structure

```
lm-rewrite-uplift/
├── core/                          # Shared library code
│   ├── model_interface.py         # Async LLM client (OpenAI-compatible API)
│   ├── model_interface_emb.py     # Async embedding model client
│   ├── answer_parser.py           # Parse structured scores from LLM responses
│   ├── prompts.py                 # All prompt templates
│   ├── utils.py                   # Inspect-ai log management utilities
│   └── vllm_inspect_provider.py   # Custom inspect-ai provider for vLLM
│
├── data_prep/                     # Dataset download and filtering
│   ├── create_local_datasets.py   # Download QA benchmarks from Hugging Face
│   ├── subset_dataset.py          # Subsample datasets to N=500 questions
│   ├── subset_hard_questions.py   # Subset questions models find challenging
│   ├── filter_rewrite.py          # Filter rewrites by quality score
│   └── filter_su.py               # Filter self-uplift rewrites by quality score
│
├── generation/                    # Question rewriting scripts
│   ├── generate_reformat.py       # Reformat questions (remove ambiguity)
│   ├── generate_answer_free_context.py  # Generate answer-free context (AFC)
│   ├── generate_afc_reformat.py   # Rewrite questions given AFC
│   ├── self_uplift.py             # Self-uplift: model rewrites then answers
│   └── get_model_embeddings.py    # Compute embeddings for questions
│
├── evaluation/                    # Inspect-ai evaluation runners
│   ├── inspect_eval_open.py       # Evaluate original/reformat questions
│   ├── inspect_eval_open_afc.py   # Evaluate questions with AFC
│   ├── inspect_eval_open_giveaway.py       # Evaluate with giveaway context
│   ├── inspect_eval_open_giveaway_afc.py   # Evaluate AFC with giveaway context
│   ├── inspect_eval_open_giveaway_afc_rewrite.py  # In-situ rewrite evaluation
│   ├── inspect_eval_open_su.py    # Evaluate self-uplift variant
│   ├── evaluate_answer_giveaway.py    # Score answer giveaway for rewrites
│   ├── evaluate_embedding.py      # Embedding similarity between question variants
│   ├── evaluate_grounding.py      # Score grounding in context
│   └── evaluate_reformat_fidelity.py  # Score rewrite fidelity to original
│
├── analysis/                      # Plotting and table generation
│   ├── plot_config.py             # Shared colors and markers for all figures
│   ├── scatterplot_per_Q_acc.py
│   ├── scatterplot_acc_vs_giveaway.py
│   ├── scatterplot_acc_vs_embedding.py
│   ├── scatterplot_acc_rewrite_vs_qAFC.py
│   ├── scatterplot_giveaway.py
│   ├── scatterplot_insitu_rewrite_results.py
│   ├── scatterplot_hle_results.py
│   ├── plot_violin_deltaAcc.py
│   ├── plot_violin_deltaAcc_insitu.py
│   ├── plot_qAFC_vs_rAFC_distribution.py
│   ├── plot_q_vs_qC_distribution.py
│   └── build_table_embeddings.py
│
├── scripts/
│   └── build_plots.sh             # Build all paper figures
│
├── .env.example                   # Required environment variables
├── models_config.json.example     # Model endpoint configuration template
└── requirements.txt               # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.12
- A running [vLLM](https://github.com/vllm-project/vllm) server (or OpenAI API access) for generation and grading

### Install Dependencies

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Configure Environment

Copy the example files and fill in your values:

```bash
cp .env.example .env
cp models_config.json.example models_config.json
```

Edit `.env` to set your server URLs and API keys. Edit `models_config.json` to configure the evaluation models (the models that answer the benchmark questions).

---

## Pipeline Overview

The full pipeline runs in five stages:

### 1. Data Preparation

Download and subset QA benchmarks (SQuAD, HotpotQA, TriviaQA, DROP, etc.) from Hugging Face:

```bash
python data_prep/create_local_datasets.py
python data_prep/subset_dataset.py          # subsample to N=500
```

### 2. Question Rewriting

Generate rewritten question variants using an LLM:

```bash
# Reformat: remove ambiguity from the original question
python generation/generate_reformat.py

# Answer-Free Context (AFC): generate a grounding context without revealing the answer
python generation/generate_answer_free_context.py

# AFC-Reformat: rewrite the question given the AFC
python generation/generate_afc_reformat.py
```

### 3. Filtering

Apply quality thresholds to remove low-quality rewrites:

```bash
python data_prep/filter_rewrite.py
```

### 4. Evaluation

Run questions through evaluation models via the `inspect-ai` framework:

```bash
# Evaluate original questions
python evaluation/inspect_eval_open.py

# Evaluate with answer-free context
python evaluation/inspect_eval_open_afc.py
```

Score quality metrics on the rewrites:

```bash
python evaluation/evaluate_answer_giveaway.py
python evaluation/evaluate_embedding.py
python evaluation/evaluate_grounding.py
```

### 5. Analysis

Generate all paper figures:

```bash
bash scripts/build_plots.sh
```

---

## Data

The processed datasets used in the paper total ~134 GB and are not included in this repository. Contact the authors or see the paper for access information.

The raw QA benchmarks can be downloaded from Hugging Face using `data_prep/create_local_datasets.py`.

---

## License

See [LICENSE](LICENSE).
