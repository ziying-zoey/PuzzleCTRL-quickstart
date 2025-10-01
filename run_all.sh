#!/usr/bin/env bash
set -euo pipefail

# ===== 可配参数 =====
DATASET="${1:-humaneval}"         # humaneval / mbpp
LIMIT="${2:-}"                    # 只导出前 N 条（可选）
BUDGETS="${BUDGETS:-2 4 6 10}"    # 扫描的工具调用预算
LAMBDA="${LAMBDA:-0.05}"          # 成本权重
OUTDIR="problems"

mkdir -p problems "${OUTDIR}"

# ===== 1) 转换数据集为 mini.jsonl =====
if [[ "${DATASET}" == "humaneval" ]]; then
  OUT_PROB="problems/humaneval_mini.jsonl"
  if [[ -n "${LIMIT}" ]]; then
    python -m runners.convert_evalplus_to_mini --dataset humaneval --out "${OUT_PROB}" --limit "${LIMIT}"
  else
    python -m runners.convert_evalplus_to_mini --dataset humaneval --out "${OUT_PROB}"
  fi
elif [[ "${DATASET}" == "mbpp" ]]; then
  OUT_PROB="problems/mbpp_mini.jsonl"
  if [[ -n "${LIMIT}" ]]; then
    python -m runners.convert_evalplus_to_mini --dataset mbpp --out "${OUT_PROB}" --limit "${LIMIT}"
  else
    python -m runners.convert_evalplus_to_mini --dataset mbpp --out "${OUT_PROB}"
  fi
else
  echo "[ERR] DATASET must be humaneval or mbpp"; exit 1
fi

echo "[OK] Problems ready -> ${OUT_PROB}"

# ===== 2) 跑 PuzzleCTRL：多预算点 =====
for B in ${BUDGETS}; do
  echo "[RUN] PuzzleCTRL  budget=${B}"
  python -m runners.run_local \
    --problems "${OUT_PROB}" \
    --method puzzlectrl \
    --budget-tool-calls "${B}" \
    --lambda-cost "${LAMBDA}"
done

# ===== 3) 跑基线（一次即可）=====
echo "[RUN] Fixed-5 baseline"
python -m runners.run_local --problems "${OUT_PROB}" --method fixed5

echo "[RUN] Greedy baseline"
python -m runners.run_local --problems "${OUT_PROB}" --method greedy

echo "[OK] All runs finished. Summaries are in ${OUTDIR}/summary_*.json"