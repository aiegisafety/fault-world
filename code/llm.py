"""统一的 LLM 调用层（DashScope / 智谱，均为 OpenAI 兼容接口）。"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_env():
    cfg = {}
    if os.path.exists(_ENV):
        for line in open(_ENV, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


CFG = _load_env()

# ENDPOINT=alt 时改用备用工作空间（不同 key 的免费额度是分开计的）
_PREFIX = "ALT" if os.environ.get("ENDPOINT") == "alt" else "DASHSCOPE"

PROVIDERS = {
    "dashscope": (os.environ.get(f"{_PREFIX}_BASE") or CFG.get(f"{_PREFIX}_BASE"),
                  os.environ.get(f"{_PREFIX}_KEY") or CFG.get(f"{_PREFIX}_KEY")),
    "zhipu": (CFG.get("ZHIPU_BASE"), CFG.get("ZHIPU_KEY")),
}

# 不接受 enable_thinking 参数的模型
NO_THINKING_PARAM = {"qwen3.8-2.4t-a95b"}

# 本进程的 token 记账（免费额度按 token 计，跑之前要能估算花销）
USAGE = {"calls": 0, "prompt": 0, "completion": 0}
_LOCK = threading.Lock()


def provider_of(model):
    """默认走阿里云工作空间端点；显式加 'zhipu:' 前缀才走智谱直连。
    （glm-5.2 等模型在阿里云上也有免费额度，不能按名字前缀分流。）"""
    return "zhipu" if model.startswith("zhipu:") else "dashscope"


def real_name(model):
    return model.split(":", 1)[1] if model.startswith("zhipu:") else model


# 硬禁用：deepseek-v4-flash 系列收费太贵（Stein 明令禁止再用）。
# 已采集的历史数据仍然有效，但不得再发起新调用。
BANNED_MODELS = {"deepseek-v4-flash", "deepseek-v4-flash-0731"}


def call(model, messages, temperature=1.0, retries=int(os.environ.get("LLM_RETRIES", "3")),
         timeout=int(os.environ.get("LLM_TIMEOUT", "70"))):
    if real_name(model) in BANNED_MODELS:
        raise RuntimeError(f"模型 {model} 已被禁用（收费太贵），不得调用")
    base, key = PROVIDERS[provider_of(model)]
    payload = {"model": real_name(model), "messages": messages,
               "temperature": temperature}
    # 思维链开关：默认关闭。开启时这些模型每次调用 60–150s，
    # 正式实验跑不动；是否开启是需要在论文里报告的固定实验参数。
    if (os.environ.get("THINKING", "0") != "1"
            and real_name(model) not in NO_THINKING_PARAM):
        payload["enable_thinking"] = False
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
                u = data.get("usage") or {}
                with _LOCK:                       # 记账：免费额度按 token 计
                    USAGE["calls"] += 1
                    USAGE["prompt"] += u.get("prompt_tokens", 0)
                    USAGE["completion"] += u.get("completion_tokens", 0)
                content = data["choices"][0]["message"].get("content")
                if content and content.strip():
                    return content
                if i == retries - 1:          # 空回复：重试，最后一次才放行
                    return content or ""
                time.sleep(2)
                continue
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            # 4xx（配额/鉴权/参数）重试无意义，立即抛出并带上服务端原文
            if 400 <= e.code < 500 and e.code != 429:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            if i == retries - 1:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            time.sleep(3 * (i + 1))
        except (urllib.error.URLError, TimeoutError):
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def parse_json(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
