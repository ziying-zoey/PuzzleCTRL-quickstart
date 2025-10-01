# PuzzleCTRL Quickstart (MVP)

An information‑gain–driven controller for budgeted code agents.
This scaffold helps you **produce quick, defensible results** for a 2‑slide deck.

## TL;DR
- Decision rule (per step): choose `a ∈ {think, execute, test, stop}` that maximizes  
  **E[ΔV | s_t, a] − λ · Cost(a)**, where `V` is a verifiable value (tests passed, etc.).
- Log **every step**: action, time, tokens (if available), ΔV (pred/obs), budget left.
- Evaluate on a **mini problem set** locally, then swap in HumanEval/MBPP via EvalPlus.

## Quick Start
```bash
cd puzzlectrl_quickstart
conda create -n puzzlectrl python=3.10 -y
conda activate puzzlectrl
pip install -U pip
pip install evalplus numpy matplotlib
# Outputs:
#  - logs/run_<timestamp>.jsonl    (per-step logs)
#  - logs/summary_<timestamp>.json (aggregate: pass@1, cost metrics)
```

## Swap in your LLM
Edit `llm/backends.py`:
- Fill `generate_candidate()` using your HF or API model.
- Optionally estimate token usage to log token‑cost.

## Use with EvalPlus (HumanEval/MBPP)
1) Install: `pip install -U evalplus`
2) Let `runners/run_evalplus.py` (TODO minimal stub) call your generator and pipe to EvalPlus.
3) Plot pass@1 vs (tokens/toolcalls/time).

## Files
- `controller.py` — MVP controller logic (E[ΔV]−λCost + stop rules).
- `envs/simple_exec.py` — Safe(ish) local execution sandbox for unit tests.
- `problems/mini.jsonl` — 3 toy problems with test assertions.
- `runners/run_local.py` — Run locally without EvalPlus to debug the pipeline.
- `llm/backends.py` — Plug your model here.
- `logs/` — Step logs & summaries.
- `slides/puzzleCTRL_2slides.pptx` — Ready‑to‑edit 2‑slide deck.

## Recommended defaults
- max_steps = 6, eps = 1e-4, tau = 0.98, lambda_cost = 0.05
- budget: wall‑time (sec) & tool_calls (count); add token budget if available.
