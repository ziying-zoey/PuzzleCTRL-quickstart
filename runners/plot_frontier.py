# -*- coding: utf-8 -*-
import json, glob, os, argparse
import matplotlib.pyplot as plt

def load_summaries(dataset_basename: str):
    rows = []
    for fp in glob.glob("logs/summary_*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        ds = j.get("dataset", "")
        if dataset_basename not in ds:
            continue
        cfg = j.get("config", {})
        agg = j.get("aggregate", {})
        rows.append({
            "file": os.path.basename(fp),
            "method": cfg.get("method", "unknown"),
            "budget_tool_calls": cfg.get("budget_tool_calls"),
            "lambda_cost": cfg.get("lambda_cost"),
            "pass": agg.get("pass_at_1_like", 0.0),
            "tools": agg.get("avg_tool_calls", 0.0),
            "time": agg.get("avg_time_sec", 0.0),
            "tokens": agg.get("avg_tokens", 0.0),
        })
    return rows

def plot_frontier(rows, dataset_tag: str, outdir="logs"):
    os.makedirs(outdir, exist_ok=True)
    methods = sorted(set(r["method"] for r in rows))

    # 工具函数：按方法分组
    def by_method(m):
        return [r for r in rows if r["method"] == m]

    # 1) pass vs tool_calls
    plt.figure()
    for m in methods:
        rs = by_method(m)
        xs = [r["tools"] for r in rs]
        ys = [r["pass"] for r in rs]
        if m == "puzzlectrl":
            # 连接成曲线（多个预算点）
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            plt.plot(xs, ys, marker="o", label=m)
        else:
            # 基线一般是单点
            plt.scatter(xs, ys, label=m, marker="x")
    plt.xlabel("avg tool calls")
    plt.ylabel("pass@1 (like)")
    plt.title(f"{dataset_tag}: pass@1 vs tool calls")
    plt.legend()
    p1 = os.path.join(outdir, f"frontier_{dataset_tag}_tools.png")
    plt.savefig(p1, dpi=160)
    print(f"[OK] {p1}")

    # 2) pass vs time
    plt.figure()
    for m in methods:
        rs = by_method(m)
        xs = [r["time"] for r in rs]
        ys = [r["pass"] for r in rs]
        if m == "puzzlectrl":
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            plt.plot(xs, ys, marker="o", label=m)
        else:
            plt.scatter(xs, ys, label=m, marker="x")
    plt.xlabel("avg time (s)")
    plt.ylabel("pass@1 (like)")
    plt.title(f"{dataset_tag}: pass@1 vs time")
    plt.legend()
    p2 = os.path.join(outdir, f"frontier_{dataset_tag}_time.png")
    plt.savefig(p2, dpi=160)
    print(f"[OK] {p2}")

    # 3) pass vs tokens
    plt.figure()
    for m in methods:
        rs = by_method(m)
        xs = [r["tokens"] for r in rs]
        ys = [r["pass"] for r in rs]
        if m == "puzzlectrl":
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            plt.plot(xs, ys, marker="o", label=m)
        else:
            plt.scatter(xs, ys, label=m, marker="x")
    plt.xlabel("avg tokens")
    plt.ylabel("pass@1 (like)")
    plt.title(f"{dataset_tag}: pass@1 vs tokens")
    plt.legend()
    p3 = os.path.join(outdir, f"frontier_{dataset_tag}_tokens.png")
    plt.savefig(p3, dpi=160)
    print(f"[OK] {p3}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval","mbpp"], required=True)
    args = ap.parse_args()
    base = "humaneval_mini.jsonl" if args.dataset=="humaneval" else "mbpp_mini.jsonl"
    rows = load_summaries(base)
    if not rows:
        print("[WARN] No summaries found for dataset:", base)
        return
    plot_frontier(rows, args.dataset)

if __name__ == "__main__":
    main()