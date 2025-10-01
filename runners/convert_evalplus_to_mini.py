# -*- coding: utf-8 -*-
"""
把 EvalPlus 的 HumanEval / MBPP 转成 PuzzleCTRL 的 mini.jsonl 输入格式：
每行一个 dict：
{
  "name": "HumanEval/0",
  "prompt": "...",
  "signature": "def foo(...):\n    pass\n",
  "entry_point": "foo",
  "tests": ["<一段可执行的测试脚本或断言>"]
}

注意：
- 对 HumanEval：其 test 往往是多行脚本（含 check() 定义）；我们保存为“一条测试脚本”的列表 [test_script]，
  运行时把整个脚本 exec 一次即可判断是否通过（通过=1/失败=0），便于在现有沙箱下跑。
- 对 MBPP：若有 test_list（断言列表），则直接保存为多条；否则回退到单条脚本。

python -m runners.convert_evalplus_to_mini \
  --dataset humaneval \
  --out problems/humaneval_mini.jsonl


python -m runners.convert_evalplus_to_mini \
  --dataset mbpp \
  --out problems/mbpp_mini.jsonl

# PuzzleCTRL
nohup python -m runners.run_local \
  --problems problems/humaneval_mini.jsonl \
  --method puzzlectrl \
  --budget-tool-calls 6 \
  --lambda-cost 0.05 \
  > logs/puzzlectrl_humaneval.log 2>&1 &

# Fixed-5
nohup python -m runners.run_local \
  --problems problems/humaneval_mini.jsonl \
  --method fixed5 \
  > logs/fixed5_humaneval.log 2>&1 &

# Greedy
nohup python -m runners.run_local \
  --problems problems/humaneval_mini.jsonl \
  --method greedy \
  > logs/greedy_humaneval.log 2>&1 &
"""
import os
import re
import json
import argparse
from typing import Dict, Any, List, Optional

def _extract_def_line(text: str) -> Optional[str]:
    """从文本中抽取第一行 def ...(: 返回 def 行"""
    if not isinstance(text, str):
        return None
    m = re.search(r"(?m)^\s*def\s+\w+\s*\(.*\):", text)
    return m.group(0) if m else None

def _ensure_signature(def_line: Optional[str]) -> Optional[str]:
    """把 def 行补成带 pass 的函数骨架"""
    if not def_line:
        return None
    return def_line.rstrip() + "\n    pass\n"

def _entry_from_signature(sig: Optional[str]) -> Optional[str]:
    if not sig:
        return None
    m = re.search(r"^\s*def\s+(\w+)\s*\(", sig)
    return m.group(1) if m else None

def _as_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(t) for t in x if str(t).strip()]
    # 单条脚本 -> 作为一条测试
    return [str(x)]

def convert_humaneval(out_path: str, limit: Optional[int] = None):
    from evalplus.data import get_human_eval_plus
    problems: Dict[str, Dict[str, Any]] = get_human_eval_plus()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for tid, prob in problems.items():
            prompt = prob.get("prompt", "")
            def_line = _extract_def_line(prompt)
            signature = _ensure_signature(def_line) or _ensure_signature(_extract_def_line(prob.get("canonical_solution",""))) or ""
            entry = prob.get("entry_point") or _entry_from_signature(signature) or ""
            # HumanEval 的 test 通常是完整脚本
            tests = _as_list(prob.get("test"))
            if not signature or not tests:
                # 跳过异常样本
                continue
            item = {
                "name": tid,
                "prompt": prompt,
                "signature": signature,
                "entry_point": entry,
                "tests": tests
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    print(f"[OK] Wrote {n} HumanEval items -> {out_path}")

# -*- coding: utf-8 -*-
import os, re, json
from typing import Dict, Any, Optional, List

def _extract_def_line(src: str) -> str:
    # 从任意文本中提取第一行 def ...(...):
    m = re.search(r"(?m)^\s*def\s+\w+\s*\([^)]*\)\s*:", src or "")
    return m.group(0) if m else ""

def _ensure_signature(defline: str) -> str:
    # 标准化：补上 pass，便于你的 simple_exec 直接替换
    return (defline + "\n    pass\n") if defline else ""

def _entry_from_signature(signature: str) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", signature or "")
    return m.group(1) if m else ""

def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]  # 把单个字符串脚本也包成列表，防止被判空

# -*- coding: utf-8 -*-
import os, re, json
from typing import Dict, Any, Optional, List

def _extract_def_line(src: str) -> str:
    m = re.search(r"(?m)^\s*def\s+\w+\s*\([^)]*\)\s*:", src or "")
    return m.group(0) if m else ""

def _ensure_signature(defline: str) -> str:
    return (defline + "\n    pass\n") if defline else ""

def _entry_from_signature(signature: str) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", signature or "")
    return m.group(1) if m else ""

# -*- coding: utf-8 -*-
import os, re, json
from typing import Dict, Any, Optional, List

def _extract_def_line(src: str) -> str:
    m = re.search(r"(?m)^\s*def\s+\w+\s*\([^)]*\)\s*:", src or "")
    return m.group(0) if m else ""

def _ensure_signature(defline: str) -> str:
    return (defline + "\n    pass\n") if defline else ""

def _entry_from_signature(signature: str) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", signature or "")
    return m.group(1) if m else ""

def _gather_tests(prob: Dict[str, Any]) -> List[str]:
    # 兼容多个可能键名；把字符串脚本也包装成列表
    tests: List[str] = []
    for k in ["assertion", "contract", "test_list", "tests", "test_code", "test", "plus_tests"]:
        v = prob.get(k, None)
        if v is None:
            continue
        if isinstance(v, list):
            tests.extend([t for t in v if isinstance(t, str) and t.strip()])
        elif isinstance(v, str):
            if v.strip():
                tests.append(v)
    # 去重
    seen, uniq = set(), []
    for t in tests:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def convert_mbpp(out_path: str, limit: Optional[int] = None, verbose: bool = True):
    from evalplus.data import get_mbpp_plus
    problems: Dict[str, Dict[str, Any]] = get_mbpp_plus()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = miss_sig = miss_tests = both = 0
    key_hist = {}

    with open(out_path, "w", encoding="utf-8") as f:
        for tid, prob in problems.items():
            # 题面（展示用）
            prompt = prob.get("prompt") or prob.get("text") or ""

            # 签名优先从 code / canonical_solution / completion 提取
            sig_src = prob.get("code") or prob.get("canonical_solution") or prob.get("completion") or prompt
            def_line = _extract_def_line(sig_src)
            signature = _ensure_signature(def_line)
            entry = prob.get("entry_point") or _entry_from_signature(signature)

            # 收集测试
            tests = _gather_tests(prob)

            # 统计测试字段分布（诊断用）
            for k in ["assertion","contract","test_list","tests","test_code","test","plus_tests"]:
                if prob.get(k) is not None:
                    key_hist[k] = key_hist.get(k, 0) + 1

            if not signature and not tests:
                both += 1
                if verbose:
                    print(f"[SKIP both] {tid}  keys={list(prob.keys())[:10]}...")
                continue
            if not signature:
                miss_sig += 1
                if verbose:
                    print(f"[SKIP sig ] {tid}  keys={list(prob.keys())[:10]}...")
                continue
            if not tests:
                miss_tests += 1
                if verbose:
                    print(f"[SKIP test] {tid}  keys={list(prob.keys())[:10]}...")
                continue

            item = {
                "name": f"MBPP/{tid}",
                "prompt": prompt,
                "signature": signature,
                "entry_point": entry,
                "tests": tests,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break

    print(f"[OK] Wrote {n} MBPP items -> {out_path}")
    print(f"[STAT] total={len(problems)}  miss_sig={miss_sig}  miss_tests={miss_tests}  both={both}")
    if key_hist:
        print("[STAT] non-empty test-like keys:", key_hist)
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--out", type=str, required=True, help="输出到 problems/*.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="只导出前 N 条（调试用）")
    args = ap.parse_args()

    if args.dataset == "humaneval":
        convert_humaneval(args.out, args.limit)
    else:
        convert_mbpp(args.out, args.limit)

if __name__ == "__main__":
    main()