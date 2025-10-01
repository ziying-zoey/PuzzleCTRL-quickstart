# -*- coding: utf-8 -*-
"""
方案A：本地 HuggingFace 推理（适配 A6000/3090 等）
- 默认模型：Qwen/Qwen2.5-Coder-7B-Instruct
- 通过环境变量 PUZZLECTRL_HF_MODEL_ID 切换模型（例如 meta-llama/Meta-Llama-3.1-8B-Instruct）
- 返回结构：{"code": <完整函数代码>, "tokens": <输入+输出 token 数>}
"""

import os
import re
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========== 你原来的备用估算器（作为兜底） ==========
def estimate_tokens_used(prompt: str, completion: str) -> int:
    # 粗略启发式：失败时退回此数
    return int(len(prompt.split()) * 0.75 + len(completion.split()))

# ========== Prompt & 文本处理工具 ==========
def build_prompt(problem: Dict, mode: str) -> str:
    """
    给大模型的提示：只产出完整函数（def ...），或仅函数体（我们会自动塞回 signature 的 pass）。
    """
    sig = (problem.get("signature") or "").strip()
    desc = (problem.get("prompt") or "").strip()
    name = problem.get("name", "")

    return f"""You are a Python code generator. Fill in the function below.

Task: {name}
Specification:
{desc}

Output ONLY valid Python code for the COMPLETE function (starting with "def ...").
If you must return only the BODY, return pure Python statements without any surrounding text.

{sig}
"""


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # 移除 ```python / ``` 包裹
        t = re.sub(r"^```(?:python)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def graft_into_signature(signature: str, body_or_func: str) -> str:
    """
    - 若返回以 def 开头：认为是完整函数，原样返回（移除围栏）。
    - 否则：视为函数体，替换 signature 中的 'pass'。
    """
    signature = signature or ""
    t = strip_code_fences(body_or_func)

    if re.search(r"^\s*def\s+\w+\s*\(", t):
        # 已是完整函数定义
        return t

    # 仅函数体 -> 塞回到 signature 的 pass
    if "pass" in signature:
        # 找到 pass 的缩进
        m = re.search(r"\n([ \t]+)pass\b", signature)
        indent = m.group(1) if m else "    "
        body_lines = [(indent + line) if line.strip() else line for line in t.splitlines()]
        body = "\n".join(body_lines) if body_lines else indent + "pass"
        return signature.replace("pass", body)

    # 找不到 pass 就原样返回 signature（兜底）
    return signature


# ========== HF 加载与生成 ==========
_HF = {"tok": None, "model": None, "model_id": None}


def _pick_dtype() -> torch.dtype:
    # A6000 支持 bfloat16，优先 bf16；否则用 fp16
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # 大多新卡都支持 bf16；用 try 更稳妥
        try:
            return torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float32


def load_hf(model_id: Optional[str] = None):
    """
    首次调用懒加载；后续复用。
    默认：Qwen/Qwen2.5-Coder-7B-Instruct
    可通过环境变量 PUZZLECTRL_HF_MODEL_ID 覆盖。
    """
    global _HF
    PUZZLECTRL_HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    if _HF["model"] is not None:
        return _HF["tok"], _HF["model"]

    model_id = model_id or os.getenv("PUZZLECTRL_HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    dtype = _pick_dtype()

    # 有些模型（如 Qwen）需要 trust_remote_code=True
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        # 需要进一步省显存可启用 4bit 量化（需安装 bitsandbytes）：
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.bfloat16 if dtype==torch.bfloat16 else torch.float16,
    )

    # pad/eos 兜底
    try:
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass
    try:
        if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    _HF.update({"tok": tok, "model": model, "model_id": model_id})
    return tok, model


def hf_generate(prompt: str,
                max_new_tokens: int = 256,
                temperature: float = 0.0) -> Tuple[str, int]:
    tok, model = load_hf()

    # === 关键：用 Chat 模板包装 ===
    messages = [
        {"role": "system", "content": "You are a helpful assistant that ONLY outputs valid Python function code."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=getattr(tok, "pad_token_id", None),
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    try:
        tokens_used = int(inputs["input_ids"].numel() + gen_ids.numel())
    except Exception:
        tokens_used = int(len(prompt.split()) * 0.75 + len(text.split()))

    return text, tokens_used
# ========== 你要替换的入口 ==========
def generate_candidate(problem: Dict, mode: str = "think") -> Dict:
    prompt = build_prompt(problem, mode)
    raw, tkn = hf_generate(prompt, max_new_tokens=256, temperature=0.0)
    code = graft_into_signature(problem.get("signature", ""), raw)
    return {"code": code, "tokens": tkn}  # 顶层 tokens，兼容现有 controller


# ========== 可选：快速自测 ==========
if __name__ == "__main__":
    # 用你的 "add_two_numbers" 这类样例做个烟囱测试
    sample = {
        "name": "add_two_numbers",
        "prompt": "Implement a function that returns the sum of two integers.",
        "signature": "def add_two_numbers(a: int, b: int) -> int:\n    pass\n",
    }
    res = generate_candidate(sample)
    print("TOKENS:", res["tokens"])
    print(res["code"])