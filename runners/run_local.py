# -*- coding: utf-8 -*-
import argparse, json, time, os, sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# === 控制器与配置 ===
from controller import ControllerCfg, PuzzleCTRL

# 如果你按我的建议在 controller.py 里实现了这两个基线，就会成功导入；
# 否则我们在 choose_runner 里会给出友好报错。
try:
    from controller import Fixed5Policy, GreedyPolicy
    _HAS_BASELINES = True
except Exception:
    _HAS_BASELINES = False

# === 执行环境：使用 SimpleExecEnv 跑（子集/全量）测试 ===
try:
    from envs.simple_exec import SimpleExecEnv
    _HAS_SIMPLE_ENV = True
except Exception as e:
    _HAS_SIMPLE_ENV = False
    print("[WARN] envs.simple_exec.SimpleExecEnv 未找到，将无法使用真实沙箱。", file=sys.stderr)
    print("       需在 controller 内置的 fallback 执行器上运行（更不安全，调试可用）。", file=sys.stderr)

class ProblemRunner:
    """
    为当前题目提供执行接口：.run(code, tests_subset_or_full) -> (ok, passed, dbg)
    tests_* 是“断言列表”，可以是全部或子集。
    """
    def __init__(self, prompt: str, entry_point: str, timeout: float = 2.0):
        self.prompt = prompt
        self.entry_point = entry_point
        self.timeout = timeout
        self.env = SimpleExecEnv() if _HAS_SIMPLE_ENV else None

    def run(self, code: str, tests: List[str]):
        if self.env is None:
            # fallback（让 controller 内部兜底；此处仅防御性）
            try:
                ns = {}
                exec(code, ns, ns)
                passed = 0
                for t in tests:
                    try:
                        exec(t, ns, ns)
                        passed += 1
                    except Exception:
                        pass
                return True, passed, {"fallback": True}
            except Exception as e:
                return False, 0, {"err": repr(e), "fallback": True}
        # SimpleExecEnv 的接口（按你之前的实现）
        # run_tests(code, prompt, tests_list, entry_point, timeout)
        ret = self.env.run_tests(code, self.prompt, tests, self.entry_point, self.timeout)
        return True, int(ret.get("pass_count", 0)), ret


# === IO 工具 ===
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def model_tag_from_env() -> str:
    # 用于区分不同模型运行
    tag = os.getenv("PUZZLECTRL_HF_MODEL_ID", "").strip()
    if not tag:
        tag = os.getenv("PUZZLECTRL_HF_LOCAL", "").strip()
    if not tag:
        return "unknown_model"
    return tag.split("/")[-1].split("\\")[-1].split(os.sep)[-1]

def choose_runner(method: str,
                  problem: Dict[str, Any],
                  cfg: ControllerCfg,
                  budget_tool_calls: Optional[int],
                  budget_time: Optional[float]):
    """
    创建“每题”runner（PuzzleCTRL 需要拿到测试列表；基线直接使用 runner.run）
    """
    # 预算配置写回 cfg
    cfg.tool_budget = budget_tool_calls
    cfg.time_budget_sec = budget_time

    prompt = problem.get("prompt", "")
    entry_point = problem.get("entry_point", "")
    tests = problem.get("tests", [])

    if method == "puzzlectrl":
        pr = ProblemRunner(prompt=prompt, entry_point=entry_point, timeout=2.0)
        # PuzzleCTRL 的构造需要 tests 与一个 runner（对接 SimpleExecEnv）
        agent = PuzzleCTRL(cfg=cfg, tests=tests, runner=pr)
        return agent, "puzzlectrl"

    if method == "fixed5":
        if not _HAS_BASELINES:
            raise RuntimeError("需要在 controller.py 中实现 Fixed5Policy")
        pr = ProblemRunner(prompt=prompt, entry_point=entry_point, timeout=2.0)
        agent = Fixed5Policy(backend=__import__("llm").backends, env=pr)
        return agent, "fixed5"

    if method == "greedy":
        if not _HAS_BASELINES:
            raise RuntimeError("需要在 controller.py 中实现 GreedyPolicy")
        pr = ProblemRunner(prompt=prompt, entry_point=entry_point, timeout=2.0)
        agent = GreedyPolicy(backend=__import__("llm").backends, env=pr, K=5)
        return agent, "greedy"

    raise ValueError(f"Unknown method: {method}")


def run_episode(agent, method: str, problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一一层调用，便于不同策略复用。
    PuzzleCTRL.run_one 接受 problem(dict)：{name/prompt/signature}
    基线 run_episode 接受 problem(dict) 同样兼容（前面 choose_runner 已适配 env）
    """
    if method == "puzzlectrl":
        # 仅提供 generate_candidate 所需信息即可
        p_in = {
            "name": problem.get("name", ""),
            "prompt": problem.get("prompt", ""),
            "signature": problem.get("signature", ""),
        }
        return agent.run_one(p_in)
    else:
        # 基线策略的 run_episode 里从 env 直接运行；传入包含 tests 的原始 problem 更稳妥
        return agent.run_episode(problem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=str, default="problems/mini.jsonl")
    ap.add_argument("--method", type=str, choices=["puzzlectrl","fixed5","greedy"], default="puzzlectrl")

    # 预算与阈值
    ap.add_argument("--budget-tool-calls", type=int, default=6)
    ap.add_argument("--budget-time", type=float, default=None)
    ap.add_argument("--lambda-cost", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--max-steps", type=int, default=12)

    # 便宜测试强度
    ap.add_argument("--quick-test-frac", type=float, default=0.3)
    ap.add_argument("--quick-test-min", type=int, default=1)

    # 随机种子
    ap.add_argument("--seed", type=int, default=0)

    # 输出目录
    ap.add_argument("--out-dir", type=str, default="logs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    problems = load_jsonl(args.problems)
    if not problems:
        print(f"[ERROR] No problems found in: {args.problems}")
        sys.exit(1)

    # 组装控制器配置
    cfg = ControllerCfg(
        lambda_cost=args.lambda_cost,
        eps=args.eps,
        tau=args.tau,
        max_steps=args.max_steps,
        time_budget_sec=args.budget_time,
        tool_budget=args.budget_tool_calls,
        quick_test_frac=args.quick_test_frac,
        quick_test_min=args.quick_test_min,
        seed=args.seed,
    )

    model_tag = model_tag_from_env()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(args.out_dir, f"run_{model_tag}_{args.method}_{ts}.jsonl")
    summ_path = os.path.join(args.out_dir, f"summary_{model_tag}_{args.method}_{ts}.json")

    # 汇总指标
    per_task = []
    pass_cnt = 0
    total_tokens = 0
    total_tools = 0
    total_time = 0.0

    with open(run_log_path, "w", encoding="utf-8") as f_log:
        for i, prob in enumerate(problems, 1):
            # 针对“每题”构造 agent（PuzzleCTRL 需要题目的 tests）
            agent, method_name = choose_runner(
                method=args.method,
                problem=prob,
                cfg=cfg,
                budget_tool_calls=args.budget_tool_calls,
                budget_time=args.budget_time,
            )

            res = run_episode(agent, method_name, prob)
            # 兼容不同策略返回
            best_value = int(res.get("best_value", 0))
            p_correct = float(res.get("p_correct", best_value / max(1, len(prob.get("tests", [])))))
            think_calls = int(res.get("think_calls", 0))
            test_calls = int(res.get("test_calls", 0))
            exec_calls = int(res.get("exec_calls", 0))
            tokens_used = int(res.get("tokens_used", res.get("tokens", 0)))
            tools_used = int(res.get("tools_used", think_calls + test_calls + exec_calls))
            elapsed = float(res.get("time", res.get("elapsed", 0.0)))
            steps = res.get("logs", [])

            # 逐题写 step 日志
            f_log.write(json.dumps({
                "task_id": prob.get("name","task_%d" % i),
                "method": method_name,
                "p_correct": p_correct,
                "best_value": best_value,
                "think_calls": think_calls, "test_calls": test_calls, "exec_calls": exec_calls,
                "tokens": tokens_used, "tools_used": tools_used, "time": elapsed,
                "steps": steps
            }, ensure_ascii=False) + "\n")

            # 汇总
            total_tests = len(prob.get("tests", []))
            pass_now = int(best_value >= total_tests and total_tests > 0)
            pass_cnt += pass_now
            total_tokens += tokens_used
            total_tools += tools_used
            total_time += elapsed

            per_task.append({
                "task_id": prob.get("name","task_%d" % i),
                "pass": bool(pass_now),
                "think_calls": think_calls, "test_calls": test_calls, "exec_calls": exec_calls,
                "time_s": elapsed, "tokens": tokens_used
            })

    # 汇总文件
    agg = {
        "dataset": os.path.basename(args.problems),
        "config": {
            "method": args.method,
            "lambda_cost": args.lambda_cost,
            "eps": args.eps, "tau": args.tau, "max_steps": args.max_steps,
            "budget_tool_calls": args.budget_tool_calls, "budget_time": args.budget_time,
            "quick_test_frac": args.quick_test_frac, "quick_test_min": args.quick_test_min,
            "seed": args.seed
        },
        "aggregate": {
            "pass_at_1_like": (pass_cnt / len(problems)) if problems else 0.0,
            "avg_tokens": (total_tokens / len(problems)) if problems else 0.0,
            "avg_tool_calls": (total_tools / len(problems)) if problems else 0.0,
            "avg_time_sec": (total_time / len(problems)) if problems else 0.0,
            "n_problems": len(problems)
        },
        "per_task": per_task
    }
    with open(summ_path, "w", encoding="utf-8") as f_sum:
        json.dump(agg, f_sum, ensure_ascii=False, indent=2)

    print("Summary:", json.dumps(agg["aggregate"], ensure_ascii=False))
    print(f"[OK] Step logs  -> {run_log_path}")
    print(f"[OK] Summary    -> {summ_path}")


if __name__ == "__main__":
    main()