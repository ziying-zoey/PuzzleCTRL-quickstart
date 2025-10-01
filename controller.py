# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Set
import time, math, random, traceback

# ==== 依赖外部模块（存在则用，不存在用兜底） ====
try:
    from envs.simple_exec import SimpleExecEnv  # 推荐的安全执行器
    _HAVE_SIMPLE_ENV = True
except Exception:
    _HAVE_SIMPLE_ENV = False

from llm.backends import generate_candidate  # 你现有的 LLM 后端
from llm import backends

import time

class Fixed5Policy:
    """
    基线策略：固定生成 5 个候选代码，再依次执行测试，取最优。
    """
    def __init__(self, backend=backends, env=None):
        self.backend = backend
        self.env = env  # ProblemRunner 会传进来

    def run_episode(self, prob: dict):
        t0 = time.time()
        cands, tokens = [], 0
        for _ in range(5):
            out = self.backend.generate_candidate(prob)
            cands.append(out["code"])
            tokens += int(out.get("usage", {}).get("tokens", 0))

        best_v, best_c, exec_calls = -1, None, 0
        for c in cands:
            ok, passed, dbg = self.env.run(c, prob["tests"])
            exec_calls += 1
            if passed > best_v:
                best_v, best_c = passed, c

        elapsed = time.time() - t0
        return {
            "best_code": best_c,
            "best_value": best_v,
            "think_calls": 5,
            "exec_calls": exec_calls,
            "test_calls": 0,
            "tokens_used": tokens,
            "logs": [],
            "elapsed": elapsed
        }


class GreedyPolicy:
    """
    基线策略：每生成一个候选就立刻执行测试，贪心保留最优。
    """
    def __init__(self, backend=backends, env=None, K=5):
        self.backend = backend
        self.env = env
        self.K = K

    def run_episode(self, prob: dict):
        t0 = time.time()
        best_v, best_c = -1, None
        tokens, exec_calls = 0, 0
        for _ in range(self.K):
            out = self.backend.generate_candidate(prob)
            tokens += int(out.get("usage", {}).get("tokens", 0))
            code = out["code"]
            ok, passed, dbg = self.env.run(code, prob["tests"])
            exec_calls += 1
            if passed > best_v:
                best_v, best_c = passed, code

        elapsed = time.time() - t0
        return {
            "best_code": best_c,
            "best_value": best_v,
            "think_calls": self.K,
            "exec_calls": exec_calls,
            "test_calls": 0,
            "tokens_used": tokens,
            "logs": [],
            "elapsed": elapsed
        }
# ------------------------
# 配置与状态
# ------------------------
@dataclass
class ControllerCfg:
    lambda_cost: float = 0.05         # 成本权重
    eps: float = 1e-4                 # 边际净增益阈值
    tau: float = 0.98                 # 满意阈值（p_correct≥tau 即停）
    max_steps: Optional[int] = 12     # 兜底步数；None 表示不设兜底
    time_budget_sec: Optional[float] = None  # 墙钟时间预算
    tool_budget: Optional[int] = 6           # 工具调用预算（think+test+execute）
    token_budget: Optional[int] = None       # 可选：token 预算
    quick_test_frac: float = 0.3      # test 动作：一次便宜测试所占比例（0~1）
    quick_test_min: int = 1           # test 动作：至少测试几个用例
    seed: int = 0                     # 随机种子（用于测试采样）

@dataclass
class Candidate:
    code: str
    # 快速测试记录：已测的测试索引集合 + 通过个数
    tested_idx: Set[int] = field(default_factory=set)
    quick_pass: int = 0
    # 全量执行结果
    full_ran: bool = False
    full_pass: int = 0
    # Beta 后验参数（对“每个测试通过率”的粗略建模）
    alpha: float = 1.0
    beta: float = 1.0

    def posterior_mean(self) -> float:
        return self.alpha / max(1e-8, self.alpha + self.beta)

    def posterior_var(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0)) if s > 2 else 0.25  # 最多 0.25

    def entropy_like(self) -> float:
        # 简单的不确定性度量：p*(1-p)，范围 [0, 0.25]
        p = self.posterior_mean()
        return p * (1.0 - p)

    def update_with_quick_result(self, success: int, total: int):
        # 用二项观测更新 Beta 后验
        self.alpha += success
        self.beta += (total - success)
        self.quick_pass += success

@dataclass
class Budget:
    tool_calls: Optional[int] = 6
    time_s: Optional[float] = None
    tokens: Optional[int] = None

@dataclass
class State:
    step: int = 0
    t0: float = field(default_factory=time.time)
    best_pass: int = 0
    total_tests: int = 1
    p_correct: float = 0.0
    tokens_used: int = 0
    tool_used: int = 0
    wall_time: float = 0.0
    last_action: str = ""
    # 候选池
    cands: List[Candidate] = field(default_factory=list)
    # 统计
    think_calls: int = 0
    test_calls: int = 0
    exec_calls: int = 0
    # EMA 用于“think”的期望增益估计
    ema_gain: float = 0.0

# ------------------------
# 执行器（兜底）
# ------------------------
def _run_code_with_tests_fallback(code: str, tests: List[str], time_limit_sec: float = 2.0) -> Tuple[bool, int, Dict[str, Any]]:
    """
    极简兜底执行器：安全性有限，仅用于无 SimpleExecEnv 的情况下。
    """
    import multiprocessing as mp

    def _worker(code_str, tests_list, q):
        ns = {}
        try:
            # 执行用户代码
            exec(code_str, ns, ns)
            # 逐条断言
            passed = 0
            for t in tests_list:
                try:
                    exec(t, ns, ns)
                    passed += 1
                except Exception:
                    pass
            q.put((True, passed, {}))
        except Exception as e:
            q.put((False, 0, {"err": repr(e)}))

    q = mp.Queue()
    p = mp.Process(target=_worker, args=(code, tests, q))
    p.start()
    p.join(timeout=time_limit_sec)
    if p.is_alive():
        p.terminate()
        return False, 0, {"err": "timeout"}
    try:
        ok, passed, dbg = q.get_nowait()
    except Exception:
        ok, passed, dbg = False, 0, {"err": "unknown"}
    return ok, passed, dbg

class _EnvAdapter:
    """
    统一一个接口：run_subset / run_full
    """
    def __init__(self):
        self.env = SimpleExecEnv() if _HAVE_SIMPLE_ENV else None

    def run_subset(self, code: str, all_tests: List[str], idx: List[int], timeout: float = 2.0) -> Tuple[bool, int, Dict[str, Any]]:
        sub = [all_tests[i] for i in idx]
        if self.env is not None:
            ret = self.env.run_tests(code, "", sub, "", timeout)
            return True, int(ret["pass_count"]), ret
        return _run_code_with_tests_fallback(code, sub, time_limit_sec=timeout)

    def run_full(self, code: str, all_tests: List[str], timeout: float = 2.0) -> Tuple[bool, int, Dict[str, Any]]:
        if self.env is not None:
            ret = self.env.run_tests(code, "", all_tests, "", timeout)
            return True, int(ret["pass_count"]), ret
        return _run_code_with_tests_fallback(code, all_tests, time_limit_sec=timeout)

# ------------------------
# 主控制器
# ------------------------
class PuzzleCTRL:
    def __init__(self, cfg: ControllerCfg, tests: List[str], runner: Optional[Any] = None):
        """
        runner: 可选自定义执行器对象（需实现 .run(code, tests) -> (ok, passed, dbg)）
        tests: 传入题目的全量测试（便宜测试会从中抽样）
        """
        self.cfg = cfg
        self.tests = tests
        random.seed(cfg.seed)
        self.env = _EnvAdapter() if runner is None else None
        self.runner = runner  # 若提供则优先使用该执行器

    # ---------- 预算与停止 ----------
    def _budget_exhausted(self, st: State) -> bool:
        # 工具预算
        if self.cfg.tool_budget is not None and st.tool_used >= self.cfg.tool_budget:
            return True
        # 时间预算
        if self.cfg.time_budget_sec is not None and st.wall_time >= self.cfg.time_budget_sec:
            return True
        # token 预算
        if self.cfg.token_budget is not None and st.tokens_used >= self.cfg.token_budget:
            return True
        # 步数兜底
        if self.cfg.max_steps is not None and st.step >= self.cfg.max_steps:
            return True
        return False

    def _stop_by_quality(self, st: State) -> bool:
        return st.p_correct >= self.cfg.tau or st.best_pass >= st.total_tests

    # ---------- gain 相关 ----------
    def _expected_gain_think(self, st: State) -> float:
        # 新候选能超过当前 best 的概率 ~ 随候选数递减
        # 可提升的上限：剩余未通过测试数
        k = max(1, len(st.cands))
        p_improve = 0.55 / math.sqrt(k)  # 可调
        delta_if = (st.total_tests - st.best_pass)
        # 引入“历史平均增益”的信用分配（让早期 think 值更高，后期衰减）
        credit = 0.5 + 0.5 * st.ema_gain
        return max(0.0, p_improve * delta_if * credit)

    def _expected_gain_execute(self, st: State) -> Tuple[float, Optional[int]]:
        # 选一个“最有希望”的候选来 full run，估计其期望通过数
        if not st.cands:
            return 0.0, None
        # 按 UCB 选择：均值 + β*标准差（更乐观）
        beta = 1.0
        best_idx, best_ucb = None, -1e9
        for i, c in enumerate(st.cands):
            if c.full_ran:
                continue
            mu = c.posterior_mean()
            var = c.posterior_var()
            ucb = mu + beta * math.sqrt(max(1e-9, var))
            if ucb > best_ucb:
                best_ucb, best_idx = ucb, i
        if best_idx is None:
            return 0.0, None
        mu = st.cands[best_idx].posterior_mean()
        exp_pass = mu * st.total_tests
        gain = max(0.0, exp_pass - st.best_pass)
        return gain, best_idx

    def _expected_gain_test(self, st: State) -> Tuple[float, Optional[int], List[int]]:
        # 对不确定性最大的候选做一小批便宜测试
        if not st.cands:
            return 0.0, None, []
        # 选择 entropy_like 最大且尚有未测用例的候选
        best_idx, best_unc = None, -1.0
        for i, c in enumerate(st.cands):
            if len(c.tested_idx) >= st.total_tests:
                continue
            unc = c.entropy_like()
            if unc > best_unc:
                best_unc, best_idx = unc, i
        if best_idx is None:
            return 0.0, None, []

        remain = [i for i in range(st.total_tests) if i not in st.cands[best_idx].tested_idx]
        batch = max(self.cfg.quick_test_min, int(math.ceil(self.cfg.quick_test_frac * st.total_tests)))
        idx = random.sample(remain, k=min(batch, len(remain)))

        # 信息价值启发式：不确定性×潜在提升×缩放
        pot = (st.total_tests - st.best_pass)
        info_value = best_unc * pot
        kappa = 0.6  # 可调
        return max(0.0, kappa * info_value), best_idx, idx

    def _cost(self, action: str, est_tests: int = 0) -> float:
        # 简单成本模型：think=1，test ~ 0.2*#cases，execute=2
        if action == "think":
            return 1.0
        if action == "test":
            return 0.2 * max(1, est_tests)
        if action == "execute":
            return 2.0
        return 0.0

    # ---------- 行动选择 ----------
    def decide(self, st: State) -> Tuple[str, Dict[str, Any]]:
        # 引导：若无候选，必须先 think
        if st.step == 0 or not st.cands:
            return "think", {"reason": "bootstrap"}

        # 停止（质量/预算）
        if self._stop_by_quality(st):
            return "stop", {"reason": "quality_satisfied"}
        if self._budget_exhausted(st):
            return "stop", {"reason": "budget_exhausted"}

        # 计算三种动作的净收益
        g_think = self._expected_gain_think(st) - self.cfg.lambda_cost * self._cost("think")
        g_exec, exec_idx = self._expected_gain_execute(st)
        g_exec = g_exec - self.cfg.lambda_cost * self._cost("execute")

        g_test, test_idx, test_subset = self._expected_gain_test(st)
        g_test = g_test - self.cfg.lambda_cost * self._cost("test", est_tests=len(test_subset))

        gains = {"think": g_think, "test": g_test, "execute": g_exec}
        best_a = max(gains, key=gains.get)
        if gains[best_a] < self.cfg.eps:
            return "stop", {"reason": "small_gain", "gains": gains}

        meta = {"gains": gains}
        if best_a == "execute":
            meta.update({"exec_idx": exec_idx})
        elif best_a == "test":
            meta.update({"test_idx": test_idx, "subset": test_subset})
        return best_a, meta

    # ---------- 外部/内部执行 ----------
    def _run_subset(self, code: str, subset_idx: List[int], timeout: float = 2.0) -> Tuple[bool, int, Dict[str, Any]]:
        if self.runner is not None:
            # 自定义 runner：一口气跑 subset（需要你 runner 自己支持）
            sub = [self.tests[i] for i in subset_idx]
            try:
                ok, passed, dbg = self.runner.run(code, sub)
            except Exception as e:
                ok, passed, dbg = False, 0, {"err": repr(e)}
            return ok, passed, dbg
        # 内置 env
        return self.env.run_subset(code, self.tests, subset_idx, timeout=timeout)

    def _run_full(self, code: str, timeout: float = 2.0) -> Tuple[bool, int, Dict[str, Any]]:
        if self.runner is not None:
            try:
                ok, passed, dbg = self.runner.run(code, self.tests)
            except Exception as e:
                ok, passed, dbg = False, 0, {"err": repr(e)}
            return ok, passed, dbg
        return self.env.run_full(code, self.tests, timeout=timeout)

    # ---------- 主流程 ----------
    def run_one(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        st = State(total_tests=len(self.tests))
        logs: List[Dict[str, Any]] = []

        # 主循环
        while True:
            # 更新时间
            st.wall_time = time.time() - st.t0
            # 决策
            action, meta = self.decide(st)
            st.last_action = action

            if action == "stop":
                logs.append({
                    "step": st.step, "action": "stop", "meta": meta,
                    "best_pass": st.best_pass, "p_correct": st.p_correct,
                    "tokens": st.tokens_used, "tools_used": st.tool_used,
                    "time": st.wall_time,
                    "budget_left": {
                        "tools": None if self.cfg.tool_budget is None else max(0, self.cfg.tool_budget - st.tool_used),
                        "time_s": None if self.cfg.time_budget_sec is None else max(0.0, self.cfg.time_budget_sec - st.wall_time),
                        "tokens": None if self.cfg.token_budget is None else max(0, self.cfg.token_budget - st.tokens_used),
                    }
                })
                break

            pred_gain = meta.get("gains", {}).get(action, 0.0)
            obs_delta = 0.0
            step_t0 = time.time()

            if action == "think":
                out = generate_candidate(problem, mode="think")
                code = out.get("code", "") or ""
                # tokens 兼容两种返回结构：usage.tokens 或 tokens
                tkn = 0
                try:
                    tkn = int(out.get("usage", {}).get("tokens", 0) or 0)
                except Exception:
                    pass
                if tkn == 0:
                    try:
                        tkn = int(out.get("tokens", 0) or 0)
                    except Exception:
                        tkn = 0
                st.tokens_used += tkn
                st.cands.append(Candidate(code=code))
                st.think_calls += 1

            elif action == "test":
                i = meta["test_idx"]
                subset = meta["subset"]
                cand = st.cands[i]
                # 选择未测用例的子集
                subset = [j for j in subset if j not in cand.tested_idx]
                if subset:
                    ok, passed, dbg = self._run_subset(cand.code, subset, timeout=2.0)
                    # 更新候选统计
                    cand.update_with_quick_result(success=passed, total=len(subset))
                    cand.tested_idx.update(subset)
                    st.test_calls += 1
                    # “观测到的增益”定义为：后验均值提升带来的期望通过数增加的近似
                    prev_mu = (cand.alpha - passed) / max(1e-8, (cand.alpha - passed) + (cand.beta - (len(subset) - passed)))
                    new_mu = cand.posterior_mean()
                    obs_delta = max(0.0, (new_mu - prev_mu) * st.total_tests)
                else:
                    # 没有可测的子集，视为 0 成本 0 增益的一步（很少发生）
                    pass

            elif action == "execute":
                i = meta["exec_idx"]
                cand = st.cands[i]
                ok, passed, dbg = self._run_full(cand.code, timeout=2.0)
                cand.full_ran, cand.full_pass = True, int(passed)
                prev_best = st.best_pass
                st.best_pass = max(st.best_pass, cand.full_pass)
                st.p_correct = st.best_pass / max(1, st.total_tests)
                obs_delta = max(0.0, st.best_pass - prev_best)
                st.exec_calls += 1

            # 计步 + 工具计数
            st.step += 1
            st.tool_used += 1
            st.wall_time = time.time() - st.t0

            # 更新 think 的 EMA 信号（让早期“think”更有吸引力，后期渐收敛）
            st.ema_gain = 0.8 * st.ema_gain + 0.2 * (st.p_correct)

            # 记录日志
            logs.append({
                "step": st.step,
                "action": action,
                "pred_gain": float(pred_gain),
                "obs_delta": float(obs_delta),
                "best_pass": int(st.best_pass),
                "p_correct": float(st.p_correct),
                "time": float(time.time() - step_t0),
                "wall_time": float(st.wall_time),
                "tokens": int(st.tokens_used),
                "tools_used": int(st.tool_used),
                "budget_left": {
                    "tools": None if self.cfg.tool_budget is None else max(0, self.cfg.tool_budget - st.tool_used),
                    "time_s": None if self.cfg.time_budget_sec is None else max(0.0, self.cfg.time_budget_sec - st.wall_time),
                    "tokens": None if self.cfg.token_budget is None else max(0, self.cfg.token_budget - st.tokens_used),
                }
            })

            # 终止检查（满分或质量阈值）
            if self._stop_by_quality(st):
                continue  # 会在下一个循环开头被 stop 捕获


        return {
            "best_value": int(st.best_pass),
            "p_correct": float(st.p_correct),
            "think_calls": int(st.think_calls),
            "test_calls": int(st.test_calls),
            "exec_calls": int(st.exec_calls),
            "tokens_used": int(st.tokens_used),
            "tools_used": int(st.tool_used),
            "time": float(st.wall_time),
            "logs": logs,
            # 便于后续对齐 EvalPlus：导出最优代码（若存在 full_ran 的最佳候选）
            "best_code": self._pick_best_code(st),
        }

    def _pick_best_code(self, st: State) -> str:
        # 优先挑 full_ran 的最高分；否则取 posterior_mean 最大的候选
        best_code, best_score = "", -1.0
        for c in st.cands:
            score = c.full_pass if c.full_ran else (c.posterior_mean() * st.total_tests)
            if score > best_score:
                best_score, best_code = score, c.code
        return best_code