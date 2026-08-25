"""跑实验前先探活：哪些模型在哪个端点上还有免费额度.

免费额度按 (账号, 模型) 计，用完会返回 403 insufficient_quota / AccessDenied。
**每次开新实验前先跑这个，把负载分散到额度多的模型上，不要盯着一个薅。**

用法: python3 probe_models.py
"""

import json
import os
import urllib.error
import urllib.request

from llm import CFG

# 禁用：deepseek-v4-flash* 收费太贵（Stein 明令禁止），不要加回来；
#       kimi-k3 / qwen3.8-max / deepseek-v4-pro 免费额度已耗尽。
BANNED = {"deepseek-v4-flash", "deepseek-v4-flash-0731", "kimi-k3",
          "qwen3.8-max", "deepseek-v4-pro-0813"}

CANDIDATES = [
    "qwen3.7-flash-2026-07-15", "qwen3.7-flash", "qwen3.7-plus",
    "qwen3.7-plus-2026-05-26", "qwen3.7-max-2026-06-08", "glm-5.2",
    "kimi-k2.7-code", "qwen3.8-27b",
    "qwen3.8-2.4t-a95b",
]
CANDIDATES = [m for m in CANDIDATES if m not in BANNED]
ENDPOINTS = [("main", CFG.get("DASHSCOPE_BASE"), CFG.get("DASHSCOPE_KEY")),
             ("alt", CFG.get("ALT_BASE"), CFG.get("ALT_KEY"))]


def probe(base, key, model):
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5}
    if model != "qwen3.8-2.4t-a95b":
        body["enable_thinking"] = False
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            json.load(r)
        return "OK"
    except urllib.error.HTTPError as e:
        d = e.read().decode("utf-8", "replace")
        for tag, short in (("insufficient_quota", "额度耗尽"),
                           ("FreeTierOnly", "额度耗尽"),
                           ("AccessDenied", "未开通/额度耗尽"),
                           ("1302", "限流")):
            if tag in d:
                return short
        return f"HTTP {e.code}"
    except Exception as e:                                  # noqa: BLE001
        return type(e).__name__


def main():
    print(f"{'model':<28}" + "".join(f"{n:>18}" for n, _, _ in ENDPOINTS))
    usable = []
    for m in CANDIDATES:
        row = []
        for name, base, key in ENDPOINTS:
            if not base or not key:
                row.append("-")
                continue
            s = probe(base, key, m)
            row.append(s)
            if s == "OK":
                usable.append((m, name))
        print(f"{m:<28}" + "".join(f"{c:>18}" for c in row))
    print("\n可用组合 (model, endpoint):")
    for m, e in usable:
        print(f"  {m}  ENDPOINT={e}")


if __name__ == "__main__":
    main()
