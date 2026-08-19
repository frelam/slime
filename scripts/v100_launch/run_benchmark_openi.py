#!/usr/bin/env python3
"""
Benchmark 评测 (Auto-benchmark-analyze) — OpenI 平台 Python 入口
================================================================
与 /slime/scripts/v100_launch/run_tool_rl_muon.py 同一套平台约定:
把「SGLang 起服务 → benchmark-diagnosis 评测+诊断 → 产物回传」整个流程
打包成 python 主程序, 供启智平台 (openi.pcl.ac.cn 云脑) 以 python 脚本
作为任务入口提交; 任务结束后自动把结果回传到平台「训练结果」供下载。

特性:
  * 自举: 若 BENCHMARK_ROOT (默认 /root/Auto-benchmark-analyze) 不存在,
    自动 clone 仓库 (gh-proxy 加速) + 建 venv + 装依赖 (阿里云镜像)。
    兼容 patch 已合入 upstream main (commit 730eecf), clone 最新即可;
    唯一仍需手工的 venv 内 patch (aime24 max_gen_toks 32768→1024,
    否则超服务上下文被 SGLang 400) 由脚本自动完成, 幂等。
  * 模型来源: MODEL_DIR env 优先 (绝对路径); 未设置时按平台"添加模型"
    挂载路径解析 (c2net pretrain_model_path: 根目录含 config.json /
    hf_base/ 子目录 / 含 .safetensors 的目录); 都没有回退
    /root/models/Qwen3-4B-tool-rl (镜像内模型)。评测用 OpenAI 兼容
    endpoint, 不需要 torch GPU 环境 — 服务用 slime env python 起 SGLang,
    评测用评测 venv python (纯 HTTP 客户端)。
  * SGLang 服务: 参数与本机 V100 实测配置一致 (TP2, torch_native 系,
    disable cuda graph, mem-fraction 0.6, tool-call-parser qwen);
    轮询 /v1/models 就绪后再评测。SGLANG_START=0 时跳过启动, 复用已有
    服务 (本地调试)。停服务: ss -tlnp 查 PID kill (launch_server 无 .py,
    pkill -f "launch_server[.]py" 匹配不到)。
  * 输出目录解析: OUTPUT_DIR env > c2net LOCAL_OUTPUT_PATH env (平台注入,
    无需 import c2net) > c2net_context.output_path > /cache/output。
    report.md / metrics.json / figures/ / eval_runs/ 直接写到输出目录,
    任务结束自动回传「训练结果」(仅保留 30 天, 可一键导出到数据集)。
  * 主动回传: 评测结束调 c2net upload_output() 主动上传一次 (平台容器内
    入口 python 装有 c2net; 本地无 c2net 自动跳过, 不影响调试)。
    OUTPUT_SYNC=0 或 OUTPUT_SYNC_UPLOAD_OUTPUT=0 关闭。
  * 所有可调参数与环境变量覆盖方式与 run_tool_rl_muon.py 一致:
    环境变量 + 命令行 KEY=VALUE (平台"运行参数"框) 均可, 已拦截不透传;
    环境变量名带空格 (如 "MODEL_DIR = xxx") 自动规范化。
  * DRY_RUN=1 / --print-command: 只打印将执行的命令与配置, 不启动。

用法:
  python /slime/scripts/v100_launch/run_benchmark_openi.py [KEY=VALUE ...]
  python /slime/scripts/v100_launch/run_benchmark_openi.py --print-command

平台提交示例 (运行参数框):
  MODEL_DIR=Qwen3-4B-tool-rl BENCHMARKS=aime24,gsm8k,ifeval,math
  (MODEL_DIR 为平台挂载模型时用挂载名, 脚本按 /tmp/pretrainmodel 解析;
   也可直接给绝对路径)

环境变量 (默认值见 CONFIG_KEYS):
  MODEL_DIR / MODEL_PARAMS / MODEL_RELEASE_DATE / BENCHMARKS / LIMIT /
  PORT / TP / CONTEXT_LENGTH / MEM_FRACTION / SGLANG_WAIT_SEC /
  BENCHMARK_ROOT / BENCHMARK_VENV / SLIME_PREFIX / SGLANG_PYTHON /
  SGLANG_START / SGLANG_LOG / CUDA_VISIBLE_DEVICES / PYPI_MIRROR /
  GIT_CLONE_URL / OUTPUT_DIR / OUTPUT_SYNC / OUTPUT_SYNC_UPLOAD_OUTPUT /
  HF_ENDPOINT / HF_ALLOW_CODE_EVAL / INSTALL_DEPS

退出码: 与 benchmark-diagnosis 一致; 编排异常统一 FATAL 退出 1。
"""

import os
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# 路径与环境
# --------------------------------------------------------------------------
BENCHMARK_ROOT = os.environ.get("BENCHMARK_ROOT", "/root/Auto-benchmark-analyze")
BENCHMARK_VENV = os.environ.get(
    "BENCHMARK_VENV", os.path.join(BENCHMARK_ROOT, ".venv")
)
SLIME_PREFIX = os.environ.get("SLIME_PREFIX", "/root/micromamba/envs/slime")
SLIME_PYTHON = os.path.join(SLIME_PREFIX, "bin", "python")
DEFAULT_MODEL_DIR = "/root/models/Qwen3-4B-tool-rl"
DEFAULT_BENCHMARKS = [
    "aime24", "bbh", "gsm8k", "humaneval", "ifeval",
    "longbench_v2", "math", "mmlu", "mmlu_pro",
]
DEFAULT_PORT = 30000

CONFIG_KEYS = {
    "MODEL_DIR", "MODEL_PARAMS", "MODEL_RELEASE_DATE", "BENCHMARKS", "LIMIT",
    "PORT", "TP", "CONTEXT_LENGTH", "MEM_FRACTION", "SGLANG_WAIT_SEC",
    "BENCHMARK_ROOT", "BENCHMARK_VENV", "SLIME_PREFIX", "SGLANG_PYTHON",
    "SGLANG_START", "SGLANG_LOG", "CUDA_VISIBLE_DEVICES", "PYPI_MIRROR",
    "GIT_CLONE_URL", "OUTPUT_DIR", "OUTPUT_SYNC", "OUTPUT_SYNC_UPLOAD_OUTPUT",
    "HF_ENDPOINT", "HF_ALLOW_CODE_EVAL", "INSTALL_DEPS",
}

PYPI_MIRROR = os.environ.get("PYPI_MIRROR", "https://mirrors.aliyun.com/pypi/simple/")
GIT_CLONE_URL = os.environ.get(
    "GIT_CLONE_URL",
    "https://gh-proxy.com/https://github.com/frelam/Auto-benchmark-analyze.git",
)


def _print(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# 平台参数解析 (与 run_tool_rl_muon.py 一致)
# --------------------------------------------------------------------------
def parse_keyvalue_args(argv: list[str]) -> list[str]:
    """拦截 KEY=VALUE 形式的已知配置键写入环境, 返回剩余参数。"""
    kept = []
    for arg in argv:
        if "=" in arg and not arg.startswith("--"):
            key, _, val = arg.partition("=")
            key = key.strip()
            if key in CONFIG_KEYS and val:
                _print(f"[run_benchmark_openi] 命令行参数 -> 配置: {key}={val}")
                os.environ[key] = val.strip()
                continue
        kept.append(arg)
    return kept


def normalize_env_spaces() -> None:
    """把带空格的变量名 (如 "MODEL_DIR = xxx") 规范化为规范名。"""
    for raw_key, raw_val in list(os.environ.items()):
        norm = raw_key.strip()
        if norm == raw_key or norm not in CONFIG_KEYS:
            continue
        if os.environ.get(norm, "").strip():
            continue
        val = raw_val.strip()
        if not val:
            continue
        _print(f"[run_benchmark_openi] WARNING: 变量名带空格 {raw_key!r} "
               f"(值={val!r}), 已规范化为 {norm}={val!r} 生效")
        os.environ[norm] = val


def resolve_output_dir() -> str:
    """解析平台输出目录: OUTPUT_DIR env > c2net LOCAL_OUTPUT_PATH env >
    c2net_context.output_path > /cache/output。"""
    env_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if env_dir:
        return env_dir
    c2net_output = os.environ.get("LOCAL_OUTPUT_PATH", "").strip()
    if c2net_output:
        _print(f"[run_benchmark_openi] c2net LOCAL_OUTPUT_PATH: {c2net_output}")
        return c2net_output
    try:
        from c2net.context import prepare  # type: ignore[import-not-found]
        ctx = prepare()
        path = getattr(ctx, "output_path", "") or ""
        if path:
            _print(f"[run_benchmark_openi] c2net 输出目录: {path}")
            return path
    except Exception as exc:  # noqa: BLE001 — 本地无 c2net 属正常
        _print(f"[run_benchmark_openi] c2net 不可用 ({exc}), 回退默认输出目录")
    return "/cache/output"


def resolve_pretrain_base() -> str:
    """平台"添加模型"挂载基目录 (c2net pretrain_model_path 约定)。"""
    return os.environ.get("PRETRAIN_MODEL_BASE", "/tmp/pretrainmodel").strip() or \
        "/tmp/pretrainmodel"


def resolve_model_dir() -> tuple[str, str, str | None]:
    """解析评测模型目录 → (path, name, chat_template_path|None)。"""
    raw = os.environ.get("MODEL_DIR", "").strip()
    if raw:
        path = raw if os.path.isabs(raw) else os.path.join(resolve_pretrain_base(), raw)
        if not os.path.isdir(path):
            _print(f"[run_benchmark_openi] ERROR: MODEL_DIR 不存在: {path}")
            sys.exit(1)
    else:
        path = _find_mounted_model()
        if path is None:
            path = DEFAULT_MODEL_DIR
        _print(f"[run_benchmark_openi] 模型目录自动解析: {path}")
    if not os.path.isfile(os.path.join(path, "config.json")):
        _print(f"[run_benchmark_openi] ERROR: {path}/config.json 不存在, "
               f"不是合法 HF 模型目录 (平台挂载模型请填 MODEL_DIR)")
        sys.exit(1)
    name = os.path.basename(path.rstrip("/"))
    chat = os.path.join(path, "chat_template.jinja")
    chat_template = chat if os.path.isfile(chat) else None
    return path, name, chat_template


def _find_mounted_model() -> str | None:
    """在平台挂载目录里找可评测的 HF 模型 (config.json + 权重文件)。

    仅含 config/tokenizer 的目录 (如 slime 训练产物的 hf_base 模板, 权重在
    iter_*/ 的 Megatron 分片里) 视为无效, 跳过。全部无效时返回 None,
    由调用方回退默认模型目录。
    """
    base = resolve_pretrain_base()
    if not os.path.isdir(base):
        return None
    candidates: list[str] = []
    for entry in sorted(os.listdir(base)):
        p = os.path.join(base, entry)
        if not os.path.isdir(p):
            continue
        if os.path.isfile(os.path.join(p, "config.json")):
            candidates.append(p)
        for sub in ("hf_base", "release"):
            s = os.path.join(p, sub)
            if os.path.isdir(s) and os.path.isfile(os.path.join(s, "config.json")):
                candidates.append(s)
    for c in candidates:
        # 权重必须真实存在 (safetensors 或 pytorch bin), 否则 SGLang 起不来
        if any(f.endswith((".safetensors", ".bin")) for f in os.listdir(c)):
            return c
    return None  # 挂载里没有完整 HF 权重 → 回退默认模型


# --------------------------------------------------------------------------
# 自举: 仓库 / venv / 依赖 / venv 内 patch
# --------------------------------------------------------------------------
def _venv_site_packages(venv: str) -> str | None:
    matches = sorted(__import__("glob").glob(os.path.join(venv, "lib", "*", "site-packages")))
    return matches[0] if matches else None


def ensure_benchmark_env() -> tuple[str, str]:
    """确保仓库 + venv + 依赖就绪, 返回 (venv_python, site_packages)。"""
    global BENCHMARK_ROOT, BENCHMARK_VENV
    BENCHMARK_ROOT = os.environ.get("BENCHMARK_ROOT", BENCHMARK_ROOT)
    BENCHMARK_VENV = os.environ.get("BENCHMARK_VENV", BENCHMARK_VENV)
    venv_python = os.path.join(BENCHMARK_VENV, "bin", "python")

    if not os.path.isdir(BENCHMARK_ROOT):
        _print(f"[run_benchmark_openi] 仓库不存在, clone: {GIT_CLONE_URL}")
        subprocess.run(["git", "clone", GIT_CLONE_URL, BENCHMARK_ROOT],
                       check=True)
    if not os.path.isfile(venv_python):
        _print(f"[run_benchmark_openi] venv 不存在, 创建: {BENCHMARK_VENV}")
        subprocess.run([sys.executable, "-m", "venv", BENCHMARK_VENV], check=True)

    site = _venv_site_packages(BENCHMARK_VENV)
    lm_eval_ok = bool(site) and os.path.isdir(
        os.path.join(site, "lm_eval")
    ) and os.path.isfile(os.path.join(site, "lm_eval", "models", "api_models.py"))

    if os.environ.get("INSTALL_DEPS", "0") == "1" or not lm_eval_ok:
        _print("[run_benchmark_openi] 安装/补装评测依赖 (阿里云镜像)...")
        pip = [venv_python, "-m", "pip", "install", "-i", PYPI_MIRROR]
        subprocess.run(pip + ["-e", f"{BENCHMARK_ROOT}[eval,plot]"], check=True)
        subprocess.run(pip + [
            "transformers", "tokenizers", "safetensors", "tenacity",
            "tiktoken", "jsonlines", "zstandard", "langdetect",
            "immutabledict",
        ], check=True)
        site = _venv_site_packages(BENCHMARK_VENV)  # 安装后重查
    if site is None:
        _print("[run_benchmark_openi] ERROR: 找不到 venv site-packages")
        sys.exit(1)

    _patch_aime24_max_gen_toks(site)
    return venv_python, site


def _patch_aime24_max_gen_toks(site: str) -> None:
    """venv 内 aime24 任务 max_gen_toks 32768→1024 (超服务上下文被 400)。幂等。"""
    p = os.path.join(site, "lm_eval", "tasks", "aime", "aime24.yaml")
    if not os.path.isfile(p):
        _print("[run_benchmark_openi] WARNING: 未找到 aime24.yaml, 跳过 patch")
        return
    with open(p, encoding="utf-8") as f:
        text = f.read()
    if "max_gen_toks: 32768" not in text:
        return  # 已 patch (或上游已修)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.replace("max_gen_toks: 32768", "max_gen_toks: 1024"))
    _print("[run_benchmark_openi] aime24.yaml: max_gen_toks 32768 -> 1024 (已 patch)")


# --------------------------------------------------------------------------
# SGLang 服务
# --------------------------------------------------------------------------
def sglang_python() -> str:
    p = os.environ.get("SGLANG_PYTHON", "").strip()
    if p:
        return p
    if os.path.isfile(SLIME_PYTHON):
        return SLIME_PYTHON
    return sys.executable  # 平台容器无 slime env 时回退当前解释器


def launch_sglang(model_dir: str, chat_template: str | None, port: int) -> subprocess.Popen:
    """后台拉起 SGLang (V100 实测参数), 返回 Popen。"""
    cmd = [
        sglang_python(), "-m", "sglang.launch_server",
        "--model-path", model_dir,
        "--tensor-parallel-size", os.environ.get("TP", "2"),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--context-length", os.environ.get("CONTEXT_LENGTH", "32768"),
        "--mem-fraction-static", os.environ.get("MEM_FRACTION", "0.6"),
        "--max-running-requests", "64",
        "--chunked-prefill-size", "512",
        "--max-prefill-tokens", "1024",
        "--attention-backend", "torch_native",
        "--sampling-backend", "pytorch",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--log-level", "info",
    ]
    if chat_template:
        cmd += ["--chat-template", chat_template]
    cmd += ["--tool-call-parser", "qwen"]
    log = os.environ.get("SGLANG_LOG", os.path.join(BENCHMARK_ROOT, "logs", "sglang_benchmark.log"))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    logf = open(log, "a", encoding="utf-8")
    _print(f"[run_benchmark_openi] 启动 SGLang: {' '.join(shlex.quote(c) for c in cmd)}")
    _print(f"[run_benchmark_openi] SGLang 日志: {log}")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=os.environ.copy())
    wait = int(os.environ.get("SGLANG_WAIT_SEC", "900"))
    _wait_ready(port, wait, log)
    return proc


def _wait_ready(port: int, timeout: int, log: str) -> None:
    """轮询 /v1/models 直到就绪; 超时打印服务日志尾部并退出。"""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    _print(f"[run_benchmark_openi] SGLang 就绪: {url}")
                    return
        except Exception:  # noqa: BLE001 — 服务未起/未就绪都在这
            pass
        if os.path.isfile(log):
            _print(f"[run_benchmark_openi] ... 等待 SGLang ({int(deadline - time.time())}s 剩余)")
        time.sleep(10)
    _print(f"[run_benchmark_openi] ERROR: SGLang 启动超时 ({timeout}s)")
    try:
        with open(log, encoding="utf-8") as f:
            _print("---- SGLang 日志尾部 ----")
            _print("\n".join(f.readlines()[-30:]))
    except OSError:
        pass
    sys.exit(1)


# --------------------------------------------------------------------------
# 评测配置生成 + 执行
# --------------------------------------------------------------------------
def build_run_config(venv_python: str, model_dir: str, model_name: str,
                     output_dir: str, port: int) -> str:
    """动态生成 benchmark-diagnosis 配置 (产物直接写输出目录)。"""
    ctx = int(os.environ.get("CONTEXT_LENGTH", "32768"))
    max_length = max(2048, ctx - 512)  # 留出生成余量, 不超服务上下文
    limit = os.environ.get("LIMIT", "").strip()
    benchmarks = os.environ.get("BENCHMARKS", "").strip() or ",".join(DEFAULT_BENCHMARKS)
    bench_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    params = os.environ.get("MODEL_PARAMS", "4.0")
    release = os.environ.get("MODEL_RELEASE_DATE", "2026-08-18")

    cfg = f"""# Auto-generated by run_benchmark_openi.py — do not edit
storage:
  db_path: {os.path.join(BENCHMARK_ROOT, 'data', 'benchmark_diagnosis.db')}
  data_dir: {os.path.join(BENCHMARK_ROOT, 'data')}

evaluation:
  harness_cmd: {os.path.join(BENCHMARK_VENV, 'bin', 'lm_eval')}
  num_fewshot: null
  batch_size: auto
  limit: {limit or 'null'}
  output_dir: {os.path.join(output_dir, 'eval_runs')}
  tokenizer: {model_dir}
  max_gen_toks: 1024
  num_concurrent: 16
  max_length: {max_length}
  confirm_run_unsafe_code: true

recommendation:
  advisor_mode: rules

run:
  mode: full
  model:
    name: {model_name}
    source: endpoint
    base_url: http://127.0.0.1:{port}/v1
    benchmarks: {bench_list}
    arch: dense
    params: {params}
    release_date: "{release}"
  output:
    dir: {output_dir}
"""
    path = os.path.join(BENCHMARK_ROOT, "logs", "benchmark_run_config.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg)
    _print(f"[run_benchmark_openi] 评测配置已写入: {path}")
    return path


def run_benchmark(venv_python: str, config_path: str) -> int:
    """跑 benchmark-diagnosis (评测 + 诊断 + 报告), 返回退出码。"""
    cmd = [os.path.join(BENCHMARK_VENV, "bin", "benchmark-diagnosis"),
           "--config", config_path, "run"]
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    _print(f"[run_benchmark_openi] 评测命令: {' '.join(shlex.quote(c) for c in cmd)}")
    _print("[run_benchmark_openi] 评测开始 (输出目录: 直接写平台输出目录, 请耐心等待)...")
    return subprocess.run(cmd, cwd=BENCHMARK_ROOT, env=env).returncode


# --------------------------------------------------------------------------
# 产物回传
# --------------------------------------------------------------------------
def write_summary(output_dir: str, model_name: str, port: int) -> None:
    lines = [
        "Auto-benchmark-analyze 评测产物 (OpenI 平台)",
        f"模型:      {model_name}",
        f"生成时间:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"SGLang:    http://127.0.0.1:{port}/v1 (TP={os.environ.get('TP', '2')}, "
        f"ctx={os.environ.get('CONTEXT_LENGTH', '32768')})",
        f"Benchmarks: {os.environ.get('BENCHMARKS', ','.join(DEFAULT_BENCHMARKS))}",
        "",
        "产物清单:",
        "  report.md        — 诊断报告 (簇判定 + 归因 + 建议)",
        "  metrics.json     — 机器可读结果",
        "  figures/         — 3 张图表",
        "  eval_runs/       — lm-eval 原始 results_*.json + samples_*.jsonl",
        "  benchmark-summary.txt — 本文件",
    ]
    with open(os.path.join(output_dir, "benchmark-summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _print(f"[run_benchmark_openi] 产物清单已写: {output_dir}/benchmark-summary.txt")


def upload_output_once() -> None:
    """调 c2net upload_output() 主动回传输出目录 (平台容器内有 c2net 才生效)。"""
    if os.environ.get("OUTPUT_SYNC_UPLOAD_OUTPUT", "1") == "0":
        _print("[run_benchmark_openi] OUTPUT_SYNC_UPLOAD_OUTPUT=0, 跳过主动回传")
        return
    try:
        from c2net.context import upload_output  # type: ignore[import-not-found]
        _print("[run_benchmark_openi] c2net upload_output() 主动回传...")
        upload_output()
        _print("[run_benchmark_openi] 回传完成 (任务详情页可下载; 任务结束还会自动回传一次)")
    except Exception as exc:  # noqa: BLE001 — 本地无 c2net 属正常
        _print(f"[run_benchmark_openi] c2net 回传跳过 ({exc}) — "
               f"平台任务结束时输出目录仍会被自动回传")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    global BENCHMARK_ROOT, BENCHMARK_VENV
    launch_args = parse_keyvalue_args(sys.argv[1:])
    normalize_env_spaces()

    # 输出目录 (先解析, 横幅要用)
    output_dir = resolve_output_dir()
    print_only = "--print-command" in launch_args or os.environ.get("DRY_RUN", "") == "1"

    venv_python, _site = ensure_benchmark_env()
    model_dir, model_name, chat_template = resolve_model_dir()

    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    ctx = os.environ.get("CONTEXT_LENGTH", "32768")
    tp = os.environ.get("TP", "2")

    # ---- 横幅 ----
    _print("============================================")
    _print("[run_benchmark_openi] Benchmark 评测 — V100 × 2 (Python 入口)")
    _print("============================================")
    _print(f"  Model:            {model_name} ({model_dir})")
    _print(f"  Benchmarks:       {os.environ.get('BENCHMARKS', ','.join(DEFAULT_BENCHMARKS))}")
    _print(f"  SGLang:           port={port}, TP={tp}, ctx={ctx}")
    _print(f"  Limit:            {os.environ.get('LIMIT', '') or '全量'}")
    _print(f"  Output dir:       {output_dir}")
    _print(f"  Benchmark root:   {BENCHMARK_ROOT}")
    _print("============================================")

    if print_only:
        cfg = build_run_config(venv_python, model_dir, model_name, output_dir, port)
        with open(cfg, encoding="utf-8") as f:
            _print("[run_benchmark_openi] DRY RUN — 评测配置:")
            _print(f.read())
        _print("[run_benchmark_openi] DRY RUN — 完成 (未启动任何进程)")
        sys.exit(0)

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            _print(f"[run_benchmark_openi] 输出目录不存在, 已创建: {output_dir}")
        except OSError as exc:
            _print(f"[run_benchmark_openi] ERROR: 无法创建输出目录 {output_dir}: {exc}")
            sys.exit(1)

    # ---- 1) SGLang 服务 ----
    proc: subprocess.Popen | None = None
    if os.environ.get("SGLANG_START", "1") == "1":
        proc = launch_sglang(model_dir, chat_template, port)
    else:
        _print(f"[run_benchmark_openi] SGLANG_START=0, 复用已有服务 (http://127.0.0.1:{port}/v1)")
        _wait_ready(port, int(os.environ.get("SGLANG_WAIT_SEC", "120")), "")

    try:
        # ---- 2) 评测 + 诊断 + 报告 (产物直接写输出目录) ----
        cfg = build_run_config(venv_python, model_dir, model_name, output_dir, port)
        rc = run_benchmark(venv_python, cfg)
        if rc != 0:
            _print(f"[run_benchmark_openi] ERROR: 评测失败 (exit {rc})")
            sys.exit(rc)

        # ---- 3) 产物清单 + 主动回传 ----
        write_summary(output_dir, model_name, port)
        upload_output_once()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            _print("[run_benchmark_openi] SGLang 服务已停止")

    _print(f"[run_benchmark_openi] 全部完成。产物在 {output_dir} "
           f"(report.md / metrics.json / figures/ / eval_runs/)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — 入口脚本统一兜底
        _print(f"[run_benchmark_openi] FATAL: {exc}")
        sys.exit(1)
