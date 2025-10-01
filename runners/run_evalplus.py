# runners/run_evalplus.py
import os, json, re
import argparse
from llm.backends import generate_candidate
from evalplus.data import get_human_eval_plus, get_mbpp_plus

def extract_signature(prompt: str):
    m = re.search(r"(?m)^\s*def\s+\w+\s*\(.*\):", prompt)
    return m.group(0) + "\n    pass\n" if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval","mbpp"], default="humaneval")
    ap.add_argument("--out", type=str, default="logs/samples_evalplus.jsonl")
    args = ap.parse_args()

    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()

    os.makedirs("logs", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for tid, prob in problems.items():
            sig = extract_signature(prob["prompt"]) or prob["prompt"].strip()
            task = {
                "name": tid,
                "prompt": "Complete the function below.",
                "signature": sig
            }
            cand = generate_candidate(task)
            out = {"task_id": tid, "completion": cand["code"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote candidates to {args.out}")

if __name__ == "__main__":
    main()