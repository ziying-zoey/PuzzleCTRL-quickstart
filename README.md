# 🧩 PuzzleCTRL — Information-Gain Control for Budgeted Code Agents 🚀

> **TL;DR**: At each step choose `a ∈ {think, execute, test, stop}` to maximize  
> \[ \mathbb{E}[\Delta V \mid s_t, a] - \lambda \cdot \text{Cost}(a) \]  
> where \(V\) is a **verifiable score** (e.g., unit tests passed).  
> PuzzleCTRL logs every step and targets **higher efficiency** (tokens / calls / time) under a **budget**.

<p align="center">
  <img src="docs/teaser.gif" alt="PuzzleCTRL teaser (drop your own GIF at docs/teaser.gif)" width="640"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/Controller-Information--Gain-green" alt="Controller: Info-Gain">
  <a href="https://github.com/evalplus/evalplus"><img src="https://img.shields.io/badge/EvalPlus-HumanEval%2FMBPP-orange" alt="EvalPlus"></a>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License">
</p>

---

## 🌱 What is PuzzleCTRL?

**PuzzleCTRL** is a lightweight, information-gain–driven controller for LLM coding agents under **budget constraints**.  
It decides when to **think**, **execute**, **test**, or **stop**, trading off expected value gain vs. execution costs.

- **Actions**: `think`, `execute`, `test`, `stop`
- **Budgets**: tool calls / wall time / (optional) tokens
- **Signals**: predicted vs. observed ΔV, pass ratio, tokens/time used
- **Outputs**: per-step logs (`.jsonl`) + summaries (`.json`)

---

## 🗂 Repository Layout

```
controller.py                      # Controller logic (E[ΔV]−λ·Cost + stop rules)
envs/simple_exec.py                # Local safe-ish execution sandbox for unit tests
problems/mini.jsonl                # 3 toy problems for quick smoke tests
llm/backends.py                    # HF local model backend (Qwen2.5-* by default)
runners/run_local.py               # Local pipeline (no EvalPlus)
runners/run_evalplus.py            # Generate EvalPlus samples (HumanEval/MBPP) w/ method flag
runners/convert_evalplus_to_mini.py# (Optional) Convert EvalPlus → mini.jsonl
scripts/plot_frontier.py           # Plot pass@1 vs. cost (tools/time/tokens)
logs/             
```

---

## ⚙️ Quick Start

### 1) Create environment
```bash
conda create -n puzzlectrl python=3.10 -y
conda activate puzzlectrl

# Install your CUDA-compatible torch first (if needed), then:
pip install -U transformers accelerate evalplus numpy matplotlib fire

# Optional: 4-bit (VRAM saving)
pip install bitsandbytes
```

### 2) (Optional) Offline-friendly setup
```bash
# Use your locally cached model; avoid network
export PUZZLECTRL_LOCAL_ONLY=1
# Swap model if you like (default is Qwen/Qwen2.5-7B-Instruct)
export PUZZLECTRL_HF_MODEL_ID="Qwen/Qwen2.5-Coder-7B-Instruct"
# Optional 4-bit
# export PUZZLECTRL_4BIT=1
```

### 3) Run on the mini set (local)
```bash
# PuzzleCTRL (budget=6, lambda=0.05)
python -m runners.run_local \
  --problems problems/mini.jsonl \
  --method puzzlectrl \
  --budget-tool-calls 6 \
  --lambda-cost 0.05

# Baselines
python -m runners.run_local --problems problems/mini.jsonl --method fixed5
python -m runners.run_local --problems problems/mini.jsonl --method greedy
```

**Outputs**
- Per-problem steps → `logs/run_<tag>_<method>_<timestamp>.jsonl`
- Summary (avg) → `logs/summary_<tag>_<method>_<timestamp>.json`  
  Fields: `pass_at_1_like`, `avg_tokens`, `avg_tool_calls`, `avg_time_sec`, …

---

## 🧠 Using Your Own LLM (HF local)

Edit **`llm/backends.py`** (already wired):

- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Switch model:
  ```bash
  export PUZZLECTRL_HF_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
  ```
- Force offline:
  ```bash
  export PUZZLECTRL_LOCAL_ONLY=1
  ```
- Return value shape:
  ```python
  {"code": "<complete def function>", "tokens": <int>}
  ```

---

## 🧪 HumanEval / MBPP via EvalPlus

> Recommended path: **generate samples** with your chosen method, then run **official scoring**.

### A) Generate samples
```bash
# HumanEval — PuzzleCTRL (budget=6)
python -m runners.run_evalplus \
  --dataset humaneval \
  --method puzzlectrl \
  --budget-tool-calls 6 \
  --lambda-cost 0.05 \
  --out logs/samples_humaneval_pctrl_B6.jsonl

# HumanEval — Greedy baseline
python -m runners.run_evalplus \
  --dataset humaneval \
  --method greedy \
  --budget-tool-calls 10 \
  --out logs/samples_humaneval_greedy.jsonl

# MBPP — PuzzleCTRL
python -m runners.run_evalplus \
  --dataset mbpp \
  --method puzzlectrl \
  --budget-tool-calls 6 \
  --out logs/samples_mbpp_pctrl_B6.jsonl
```

### B) Official scoring (base + plus)
```bash
evalplus.evaluate --dataset humaneval --samples logs/samples_humaneval_pctrl_B6.jsonl
evalplus.evaluate --dataset humaneval --samples logs/samples_humaneval_greedy.jsonl

evalplus.evaluate --dataset mbpp --samples logs/samples_mbpp_pctrl_B6.jsonl
```

> The controller uses local tests (incl. `assertion/contract` for MBPP) **only to inform decisions**.  
> The **official score** is always from `evalplus.evaluate`.

---

## 📊 Example Results (HumanEval subset, n=50)

| Method      | Budget (B) | pass@1_like | Avg_Tokens | Avg_Tool_Calls | Avg_Time_s |
|-------------|------------|-------------|------------|----------------|------------|
| PuzzleCTRL  | 2          | **1.00**    | **271.16** | **2.00**       | **1.85**   |
| PuzzleCTRL  | 4          | **1.00**    | **271.16** | **2.00**       | **1.85**   |
| PuzzleCTRL  | 6          | **1.00**    | **271.16** | **2.00**       | **3.84**   |
| PuzzleCTRL  | 10         | **1.00**    | **271.16** | **2.00**       | **1.86**   |
| Fixed-5     | 5          | **1.00**    | 1355.80    | 10.00          | 8.73       |
| Greedy      | 10         | **1.00**    | 1355.80    | 10.00          | 8.81       |

**Takeaway**: On this easy subset, all reach pass@1=1.0,  
but **PuzzleCTRL** uses **~5× fewer tokens** and **~5× fewer tool calls**, with lower latency. ✅

> *Note*: `pass@1_like` is a local open-test proxy; use `evalplus.evaluate` for official base/plus scores.

---

## 📈 Plot Cost–Performance Frontiers

```bash
# Reads logs/summary_*.json and plots:
#   pass@1 vs tool calls / time / tokens
python -m scripts.plot_frontier --dataset humaneval
# → logs/frontier_humaneval_tools.png, _time.png, _tokens.png
```

---

## 🧭 Roadmap

- **Evaluation**: Full **HumanEval (164)** / **MBPP (974)** with base/plus; frontiers + AUC-like summaries
- **Baselines**: CoT-only, Always-execute (PAL-like), Self-refine, Coverage-threshold stop, Bandits (UCB/TS)
- **Controller**: Bayesian stopping (credible intervals), minimal-patch `refine`, contextual bandits / RL
- **Datasets**: APPS (Intro/Interview), CodeContests subsets

---

## 🛠 Troubleshooting

<details>
<summary>🤖 HF offline / mirrors / timeouts</summary>

- Prefer cached weights:
  ```bash
  export PUZZLECTRL_LOCAL_ONLY=1
  ```
- If a mirror is misbehaving, unset mirror envs:
  ```bash
  unset HF_ENDPOINT
  unset HF_HUB_ENABLE_HF_TRANSFER
  ```
</details>

<details>
<summary>💾 CUDA OOM</summary>

- Use 4-bit:
  ```bash
  pip install bitsandbytes
  export PUZZLECTRL_4BIT=1
  ```
- Try a smaller model; keep `device_map="auto"` (default).
</details>

<details>
<summary>🔒 Sandbox errors</summary>

- `envs/simple_exec.py` is safe-ish for local debugging.  
  For stricter isolation, consider Docker or jailed subprocess with time/mem limits.
</details>

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgments

- **EvalPlus** for HumanEval/MBPP harness and plus tests.  
- **Qwen** and other model providers under their respective licenses.

---

## 🌟 Star this repo if you find it helpful!  
Made with ❤️ and a lot of ☕.
