import time, types, traceback
from typing import List, Tuple

def run_code_with_tests(code: str, tests: List[str], time_limit_sec: float = 2.0) -> Tuple[bool, int, str]:
    """
    Executes `code` in an empty module, then runs each test (string assert).
    Returns: (all_passed, passed_count, debug_log)
    """
    start = time.time()
    debug = []
    mod = types.ModuleType("candidate")
    # Restrict builtins a bit (still not fully secure; for serious eval use a sandbox)
    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "sum": sum,
        "abs": abs, "enumerate": enumerate, "list": list, "dict": dict, "set": set,
        "True": True, "False": False, "int": int, "float": float, "str": str, "print": print
    }
    mod.__dict__["__builtins__"] = safe_builtins

    try:
        exec(code, mod.__dict__)
    except Exception as e:
        return (False, 0, "compile_error: " + traceback.format_exc())

    passed = 0
    for t in tests:
        if time.time() - start > time_limit_sec:
            debug.append("timeout")
            break
        try:
            exec(t, mod.__dict__)
            passed += 1
            debug.append(f"PASS: {t}")
        except Exception as e:
            debug.append(f"FAIL: {t} -> {e}")
    all_ok = (passed == len(tests))
    return (all_ok, passed, "\n".join(debug))
