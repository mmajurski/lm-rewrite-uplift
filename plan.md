# LM-Rewrite-Uplift Codebase Cleanup Plan

## Project Context

This is a research codebase for a paper on **Query Disambiguation via Answer-Free Context** — investigating whether rewriting questions to be more explicit (less ambiguous) improves LLM accuracy on open-ended QA. The pipeline is:

1. **Data prep** — Download QA benchmarks (SQuAD, HotpotQA, TriviaQA, etc.), subset them
2. **Rewriting** — Use LLMs to rewrite questions in several ways (reformat, answer-free context, self-uplift)
3. **Filtering** — Quality-gate rewrites by similarity/giveaway thresholds
4. **Evaluation** — Run original vs. rewritten questions through LLMs via the `inspect-ai` framework
5. **Analysis** — Plot results, build tables

There are ~38 Python files (all in root), ~134GB of data directories, and a LaTeX paper.

---

## Phase 1: Security & Sensitivity Scrub (Do First)

The codebase has internal NIST infrastructure details hardcoded across 16 files. This must be cleaned before any public release.

### 1.1 Remove Hardcoded Server Names & URLs
**Files affected:** all `inspect_eval_open_*.py` (6 files), `model_interface.py`, `model_interface_emb.py`, `generate_*.py` (3 files), `evaluate_*.py` (4 files), `self_uplift.py`, `get_model_embeddings.py`

**Actions:**
- [ ] In `model_interface.py`: replace the `translate_remote()` function's hardcoded NIST hostnames (`pn131285.nist.gov`, `pn131274.nist.gov`, etc.) with a config-driven approach (dict loaded from env or a config file)
- [ ] In `model_interface_emb.py`: same treatment
- [ ] In all `inspect_eval_open_*.py`: remove commented-out server blocks and hardcoded `GRADER_MODEL_BASE_URL` values — make these configurable via env vars or CLI args
- [ ] In all `generate_*.py`, `evaluate_*.py`, `self_uplift.py`: audit for any hardcoded URLs passed to model constructors (these take `remote` as a parameter, so the real fix is in `model_interface.py`)
- [ ] Remove any references to specific NIST machine names (sierra, oscar, papa, echo, foxtrot, golf, hotel, redwing)

### 1.2 Audit for Credentials
- [ ] Confirm `.env` is in `.gitignore` (it is)
- [ ] Create `.env.example` listing required variables without values
- [ ] Search for any accidentally committed API keys or tokens
- [ ] Check git history for leaked secrets (if public, may need to rotate keys)

---

## Phase 2: Delete Dead Weight

### 2.1 Files to Delete
- [ ] `tmp.py` — scratch file, entirely commented-out test snippets
- [ ] Commented-out code blocks across all files (there are many: old grader configs, debug prints, alternate dataset paths)

### 2.2 Clean Up Git-Tracked Deletions
The following files show as deleted in git status but haven't been committed:
- [ ] `regrade_logs.py`, `regrade_logs2.py`
- [ ] `runner.sh`, `runner2.sh`, `runner3.sh`
- [ ] `vllm_emb_e5_7B.sh`, `vllm_gpt-120b.sh`, `vllm_gpt-20b.sh`
- [ ] Stage these deletions and commit

### 2.3 Data Directories
Per your note: ignore everything in `data*/` and `source_data/` directories.
- [ ] Ensure `.gitignore` excludes all data directories (it already excludes most)
- [ ] Do NOT include these in the public repo — they total ~134GB
- [ ] Add a section to README explaining how to obtain/generate the data

---

## Phase 3: Organize Files Into Directories

The current structure is 38 `.py` files dumped in root. The goal is not to turn this into a pip-installable package (it's research code), but to group files logically so a reader can navigate them.

### 3.1 Proposed Structure
```
lm-rewrite-uplift/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── core/                          # Shared library code
│   ├── model_interface.py         # LLM API client (async)
│   ├── model_interface_emb.py     # Embedding model client
│   ├── answer_parser.py           # Response parsing utilities
│   ├── prompts.py                 # All prompt templates
│   ├── utils.py                   # Log management helpers
│   └── vllm_inspect_provider.py   # Custom inspect-ai provider
│
├── data_prep/                     # Dataset download, subsetting, filtering
│   ├── create_local_datasets.py
│   ├── subset_dataset.py
│   ├── subset_hard_questions.py
│   ├── filter_rewrite.py
│   └── filter_su.py
│
├── generation/                    # Question rewriting scripts
│   ├── generate_reformat.py
│   ├── generate_answer_free_context.py
│   ├── generate_afc_reformat.py
│   ├── self_uplift.py
│   └── get_model_embeddings.py
│
├── evaluation/                    # Quality scoring & inspect-ai runners
│   ├── evaluate_reformat_fidelity.py
│   ├── evaluate_answer_giveaway.py
│   ├── evaluate_embedding.py
│   ├── evaluate_grounding.py
│   ├── inspect_eval_open.py       # Consolidated (see Phase 4)
│   └── inspect_eval_open_variants.py  # Or kept as separate files
│
├── analysis/                      # Plotting and table generation
│   ├── plot_config.py             # Shared colors, markers, styles (new)
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
├── scripts/                       # Shell scripts
│   └── build_plots.sh
│
└── paper/                         # (unchanged)
```

### 3.2 Actions
- [ ] Create `core/`, `data_prep/`, `generation/`, `evaluation/`, `analysis/`, `scripts/` directories
- [ ] Move files to their new locations
- [ ] Update all internal imports (most are simple: `import answer_parser` → `from core import answer_parser`, etc.)
- [ ] Update `sys.path` hacks if any exist
- [ ] Update `build_plots.sh` paths
- [ ] Test that scripts still run after the move

---

## Phase 4: Reduce Duplication

### 4.1 Consolidate `inspect_eval_open_*.py` (6 files → 1-2 files)
These 6 files are nearly identical (~350 lines each). The differences are:
- Which field from the JSON is used as the question (orig question, reformat, AFC, self-uplift)
- Whether giveaway context is prepended
- `max_connections` and `timeout` values

**Actions:**
- [ ] Create a single `inspect_eval_open.py` that accepts CLI args or a config to select the variant
- [ ] The `@solver` function and grading logic are identical — factor them out
- [ ] The `main()` function's dataset loading loop is the only meaningful difference — parameterize it
- [ ] Keep variant behavior as named modes (e.g., `--mode orig|reformat|afc|giveaway|su`)

### 4.2 Extract Shared Plotting Config
Every `scatterplot_*.py` and `plot_*.py` file copy-pastes the same 34-element `plot_colors` list and `plot_markers` list.

**Actions:**
- [ ] Create `analysis/plot_config.py` with shared `COLORS`, `MARKERS`, and common styling
- [ ] Update all plotting scripts to `from plot_config import COLORS, MARKERS`
- [ ] Do NOT try to merge the plotting scripts themselves — each generates a specific figure for the paper, and merging them would make them harder to understand

### 4.3 Generation Scripts
`generate_reformat.py`, `generate_answer_free_context.py`, `generate_afc_reformat.py` follow the same pattern. They're different enough in prompt construction that merging is risky.

**Action:**
- [ ] Leave as separate files but extract the shared pattern (load dataset → build prompts → call model → parse → save) into a helper if it's clean to do so. If not, leave them — 3 similar files is fine for research code.

---

## Phase 5: Code Cleanup Within Files

### 5.1 Remove Commented-Out Code
Across the codebase there are many blocks of commented-out alternative configurations, old imports, and debug code.

**Actions:**
- [ ] Remove commented-out grader model configurations in all `inspect_eval_open_*.py` files
- [ ] Remove `# TODO remove` debug lines
- [ ] Remove commented-out `import` statements
- [ ] Remove dead code paths that are fully commented out
- [ ] Keep comments that explain *why* something is done a certain way

### 5.2 Clean Up `model_interface.py`
- [ ] Replace the long if/elif chain in `translate_remote()` with a dictionary lookup
- [ ] Remove NIST-specific hostnames (see Phase 1)
- [ ] Add docstrings to `SglModelAsync` class and its methods

### 5.3 Clean Up `prompts.py`
- [ ] Add a module docstring explaining what each prompt is for
- [ ] Add a brief comment above each prompt constant explaining its purpose
- [ ] Verify all prompts are actually used (remove any orphaned ones)

### 5.4 Clean Up `answer_parser.py`
- [ ] This is 524 lines of regex-heavy parsing — add docstrings and brief comments explaining each parser function
- [ ] Remove any unused parsing functions

### 5.5 Clean Up `filter_rewrite.py` and `filter_su.py`
- [ ] These have hardcoded `ifp` paths and `fldrs` lists commented in/out — make these CLI arguments or at minimum document what to change

---

## Phase 6: Documentation

### 6.1 Rewrite README.md
The current README is a mix of setup instructions and research planning notes. Replace with:

- [ ] **Project title and one-paragraph description** of the research
- [ ] **Paper citation** (bibtex)
- [ ] **Repository structure** (the directory layout from Phase 3)
- [ ] **Setup instructions** (clean up the existing ones)
- [ ] **Pipeline overview** — brief description of each step: data prep → rewriting → filtering → evaluation → analysis
- [ ] **How to reproduce results** — which scripts to run in what order
- [ ] **Data** — where to get the datasets (or note they're available on request)
- [ ] **License**

### 6.2 Add Module-Level Docstrings
- [ ] Every `.py` file should have a 1-3 line module docstring at the top explaining what it does
- [ ] This is the single highest-value documentation improvement for research code

### 6.3 Add Docstrings to Key Functions
Focus on the library code in `core/`:
- [ ] `model_interface.py`: `SglModelAsync.__init__`, `generate`
- [ ] `answer_parser.py`: each `parse_*` function
- [ ] `prompts.py`: module-level description of each prompt's purpose

Don't over-document scripts — a module docstring is sufficient.

---

## Phase 7: Dependency & Environment Cleanup

### 7.1 Create `requirements.txt`
- [ ] Generate from current `.venv`: `pip freeze > requirements.txt`
- [ ] Or create a minimal one listing only direct dependencies:
  ```
  openai
  jsonpickle
  datasets
  matplotlib
  scikit-learn
  python-Levenshtein
  pandas
  numpy
  scipy
  python-dotenv
  inspect-ai
  httpx
  transformers  # if needed for embeddings
  ```
- [ ] Pin major versions for reproducibility

### 7.2 Update `.gitignore`
- [ ] Add standard Python patterns: `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`
- [ ] Ensure all data directories are excluded
- [ ] Add `paper/*.aux`, `paper/*.log`, etc. (LaTeX build artifacts)
- [ ] Remove entries for things that don't exist

---

## Phase 8: Final Polish

### 8.1 Add LICENSE
- [ ] Choose and add a license file (check with your institution — NIST typically uses public domain or specific government licenses)

### 8.2 Formatting Pass
- [ ] Run `black` or `autopep8` across all files for consistent formatting
- [ ] Run `isort` to sort imports
- [ ] This is cosmetic but makes the code look professional

### 8.3 Final Review Checklist
- [ ] No NIST hostnames or internal URLs anywhere in the code
- [ ] No API keys or credentials in any tracked file
- [ ] No `tmp.py` or scratch files
- [ ] No 4800-line files with hardcoded data
- [ ] Every file has a module docstring
- [ ] README explains how to reproduce results
- [ ] All scripts can find their imports after the directory reorganization
- [ ] `.gitignore` is complete
- [ ] License is present

---

## What NOT to Do

This is research code accompanying a paper. The goal is **presentable and understandable**, not production-grade. Avoid:

- **Don't create a pip-installable package** (`src/` layout, `setup.py`, `pyproject.toml`) — overkill for a paper repo
- **Don't write a test suite** — there's no CI, no one will run tests, and the real validation is the paper results
- **Don't add type hints everywhere** — focus effort on docstrings and comments instead
- **Don't create an elaborate config system** — simple env vars and CLI args are fine
- **Don't merge all plotting scripts into one** — each generates a specific paper figure and is easier to understand standalone
- **Don't create `CONTRIBUTING.md`** or `docs/` directory — a good README is sufficient
- **Don't refactor working code for elegance** — if it produced the paper's results, it works

---

## Execution Order

1. **Phase 1** (Security scrub) — must be done before anything goes public
2. **Phase 2** (Delete dead weight) — quick wins, reduces noise
3. **Phase 3** (Organize into directories) — the big structural change
4. **Phase 4** (Reduce duplication) — do after files are in place
5. **Phase 5** (Code cleanup) — file-by-file cleanup
6. **Phase 6** (Documentation) — README and docstrings
7. **Phase 7** (Dependencies) — requirements.txt and .gitignore
8. **Phase 8** (Final polish) — formatting, license, review
