import os
import json
import threading
import asyncio
import nest_asyncio
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

# ADK / Agent imports
from google.adk.tools.mcp_tool.mcp_toolset import SseConnectionParams
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from dp.agent.adapter.adk import CalculationMCPToolset

# Tools used by the agent
from vasp_function import (
    read_vasp_pdf,
    write_vasp_report,
    analyze_vasprun_all,
    search_poscar_template,
    write_poscar,
    write_vasp_config,
    show_vasp_config,
    rewrite_vasp_config,
)
from utils import ask_human_for_advice, show_task_status

# Allow nested event loops for safety in some environments
nest_asyncio.apply()
load_dotenv()

# Agent prompt from agent.py
agent_prompt = '''
你是一个集文献阅读、材料建模、VASP 配置、任务执行、结果分析和报告撰写于一体的智能科研助理。

你的整体目标是：根据用户提供的材料体系或文献，**自动生成结构、配置并提交 VASP 任务，最终分析结果并生成一份标准报告**。
---
请你按顺序完成以下任务，不要跳步，不要遗漏任何一步。
请注意，提交任务前一定要等人类反馈！！！
## 🧠 工作总流程如下：

1. **获取任务信息**
    - 向用户提问：请提供论文路径（PDF）或目标化学式。
    - 使用 `read_vasp_pdf` 工具获取论文内容（无需回复内容，只提取信息）。

2. **结构构建与确认**
    - 使用 `search_poscar_template` 生成 POSCAR 模板。
    - ***
    对POSCAR模板进行原子替换(不再需要search_poscar_template)，确保结构中的所有原子种类、数量和分布都严格符合输入化学式人类希望复现的化学式，请认真完成这最重要的一步。
    比如：
    输入化学式：Sr5Ca3Fe8O24
    人类希望复现的化学式：Sr5Ca3Fe8O24
    那么POSCAR中应该包含Sr,Ca,Fe,O四种原子，且原子数量分别为5,3,8,24，下面的坐标需要根据化学式进行替换，确保符合晶体结构，且元素数目与化学式严格一致。
    ***。
    - 使用ask_human_for_advice 向用户询问原子替换后 POSCAR 文件内容如下,再根据用户反馈进行修改，直到用户满意为止，再进行ti：
      ```
      原子替换后 完整POSCAR 文件内容如下：
      [内容]

      请问你有什么修改意见？
      ```
    - 等待用户确认并接收修改建议。

3. **生成计算配置并检查**
    - 调用 `write_poscar` 写入 POSCAR，注意这一步需要你需要将原子替换后的，完全符合POSCAR格式的str输入到函数中（注意顶行是化学式）。
    - 根据材料体系命名一个路径 `calcdir`，例如 "LaFeO3"
    - 使用 `write_vasp_config` 写入到calcdir中 生成 INCAR, KPOINTS, POTCAR。
    - 若有缺失，重新生成，确保生成成功。


4. **VASP 提交与监听**
    - 调用 `show_vasp_config` 获取INCAR 文件内容：
    - 使用ask_human_for_advice 将INCAR文件内容展示给用户，并询问用户是否可以提交任务：
      ```
      以下是 INCAR 文件内容：
      [内容]
      是否可以继续提交任务？
      ```

    - 等到用户确认后，使用 `vasp_job` 提交任务，**注意不要擅自主动提前提交任务**，并监听结果，返回 `xml_path`。

5. **结果分析与报告撰写**
    - 使用 `analyze_vasprun_all(xml_path)` 分析任务结果。
    - 生成一份标准化报告，内容包括：
      - 程序与平台信息
      - INCAR 设置摘要
      - K 点设置与自动化情况
      - 结构、力、错误信息
      - 其他重要输出
    - 使用 `write_vasp_report(report_str)` 写入文件。

---

## 🔧 工具使用说明

### 文献阅读工具
- `read_vasp_pdf(pdf_path)`: 读取PDF文献内容，提取材料信息

### 结构建模工具
- `search_poscar_template(formula)`: 搜索POSCAR模板
- `write_poscar(poscar_content, calcdir)`: 写入POSCAR文件

### VASP配置工具
- `write_vasp_config(calcdir, formula, functional, kpoints, encut)`: 生成VASP配置文件
- `show_vasp_config(calcdir)`: 显示当前配置
- `rewrite_vasp_config(calcdir, new_config)`: 重写配置

### 任务执行工具
- `vasp_job(calcdir)`: 提交VASP计算任务

### 结果分析工具
- `analyze_vasprun_all(xml_path)`: 分析vasprun.xml结果
- `write_vasp_report(report_content)`: 生成分析报告

### 交互工具
- `ask_human_for_advice(question)`: 向用户询问意见
- `show_task_status()`: 显示当前任务状态

---

## 📋 注意事项

1. **严格按流程执行**：不要跳步，每一步都要完成
2. **等待用户确认**：提交任务前必须得到用户同意
3. **原子数量匹配**：POSCAR中的原子数量必须与化学式严格一致
4. **错误处理**：遇到错误时要重试或寻求用户帮助
5. **文件路径**：注意文件路径的正确性，使用相对路径

记住：你的目标是帮助用户完成完整的VASP计算流程，从文献分析到最终报告生成。
'''

app = Flask(__name__)

# 配置 Flask，避免计算文件变化导致应用重启
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# -----------------------------
# Project root and safe paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  
ALLOWED_ROOTS = {
    "tmp": os.path.join(PROJECT_ROOT, "tmp"),
    "server": os.path.join(PROJECT_ROOT, "server"),
    ".": PROJECT_ROOT,  # 允许访问项目根目录（用于PDF扫描）
}


def secure_path(path: str) -> str:
    """Ensure the requested path stays within allowed roots (tmp/server/.)."""
    norm = os.path.normpath(path).lstrip(os.sep)

    # 特殊处理根目录
    if norm == "." or norm == "":
        return PROJECT_ROOT

    # pick which root it belongs to
    for key, root in ALLOWED_ROOTS.items():
        if norm.startswith(key + os.sep) or norm == key:
            if key == ".":
                # 对于根目录，只允许直接访问，不允许子目录遍历
                if norm == ".":
                    return PROJECT_ROOT
                else:
                    continue
            full = os.path.join(PROJECT_ROOT, norm)
            abs_root = os.path.abspath(root)
            abs_full = os.path.abspath(full)
            if os.path.commonpath([abs_full, abs_root]) == abs_root:
                return abs_full
    raise ValueError("Invalid or disallowed path")


# -----------------------------
# Agent initialization (one-time) with background event loop
# -----------------------------
_init_lock = threading.Lock()
_initialized = False
_runner: Runner | None = None
_session_service: InMemorySessionService | None = None
_vasp_agent: LlmAgent | None = None
_background_loop: asyncio.AbstractEventLoop | None = None
_background_thread: threading.Thread | None = None


def _make_executor_and_storage():
    home_dir = os.path.expanduser("~")
    bohr_path = os.path.join(home_dir, ".bohrium")
    os.environ["PATH"] = os.environ.get("PATH", "") + f":{bohr_path}"

    # Use current working directory to match agent.py behavior exactly
    current_dir = os.getcwd()
    bohr_executor = {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": os.environ.get("BOHRIUM_USERNAME"),
                "password": os.environ.get("BOHRIUM_PASSWORD"),
                "project_id": int(os.environ.get("BOHRIUM_PROJECT_ID")) if os.environ.get("BOHRIUM_PROJECT_ID") else 0,
                "input_data": {
                    "image_name": "registry.dp.tech/dptech/vasp:5.4.4",
                    "job_type": "container",
                    "platform": "ali",
                    "scass_type": "c32_m32_cpu",
                },
            },
        },
        "DEFAULT_FORWARD_DIR": [f"{current_dir}/tmp"],
    }
    bohr_storage = {
        "type": "bohrium",
        "username": os.environ.get("BOHRIUM_USERNAME"),
        "password": os.environ.get("BOHRIUM_PASSWORD"),
        "project_id": int(os.environ.get("BOHRIUM_PROJECT_ID")) if os.environ.get("BOHRIUM_PROJECT_ID") else 0,
    }
    return bohr_executor, bohr_storage


def _run_background_loop():
    """Run the background event loop in a separate thread."""
    global _background_loop
    _background_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_background_loop)
    _background_loop.run_forever()


async def _init_agent_async():
    """Initialize agent components in the background event loop."""
    global _runner, _session_service, _vasp_agent

    bohr_executor, bohr_storage = _make_executor_and_storage()

    vasp_tools = await CalculationMCPToolset(
        connection_params=SseConnectionParams(url="http://localhost:8000/sse"),
        executor=bohr_executor,
        storage=bohr_storage,
    ).get_tools()

    # Wrap python functions as ADK tools
    _read_vasp_pdf = FunctionTool(func=read_vasp_pdf)
    _write_vasp_report = FunctionTool(func=write_vasp_report)
    _analyze_vasprun_all = FunctionTool(func=analyze_vasprun_all)
    _search_poscar_template = FunctionTool(func=search_poscar_template)
    _write_poscar = FunctionTool(func=write_poscar)
    _write_vasp_config = FunctionTool(func=write_vasp_config)
    _show_vasp_config = FunctionTool(func=show_vasp_config)
    _rewrite_vasp_config = FunctionTool(func=rewrite_vasp_config)
    _ask_human_for_advice = FunctionTool(func=ask_human_for_advice)
    _show_task_status = FunctionTool(func=show_task_status)

    _vasp_agent = LlmAgent(
        model=LiteLlm(model="openrouter/openai/gpt-4o"),
        name="vasp_agent",
        description=(
            "A phd who is good at using VASP to calculate the properties of materials."
        ),
        instruction=agent_prompt,
        tools=[
            _show_task_status,
            _ask_human_for_advice,
            _read_vasp_pdf,
            _write_poscar,
            _write_vasp_config,
            _show_vasp_config,
            _rewrite_vasp_config,
            _analyze_vasprun_all,
            _write_vasp_report,
            _search_poscar_template,
            *vasp_tools,
        ],
    )

    _session_service = InMemorySessionService()

    _runner = Runner(
        app_name="adk_agent_samples",
        agent=_vasp_agent,
        session_service=_session_service,
    )


def ensure_initialized():
    """Initialize the agent with a background event loop."""
    global _initialized, _background_thread, _background_loop
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        # Start background event loop in a separate thread
        _background_thread = threading.Thread(target=_run_background_loop, daemon=True)
        _background_thread.start()

        # Wait for the loop to be ready
        import time
        while _background_loop is None:
            time.sleep(0.01)

        # Initialize agent in the background loop
        future = asyncio.run_coroutine_threadsafe(_init_agent_async(), _background_loop)
        future.result()  # Wait for completion

        _initialized = True


# -----------------------------
# Helpers
# -----------------------------

def _collect_events_sync(session_id: str, user_id: str, text: str) -> List[Dict[str, Any]]:
    """Run one turn and collect text parts using the background event loop."""
    content = types.Content(role="user", parts=[types.Part(text=text)])

    async def _run():
        assert _runner is not None
        events = _runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=content,
        )
        out: List[Dict[str, Any]] = []
        async for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        out.append({
                            "role": event.content.role,
                            "text": part.text,
                        })
        return out

    # Use the background event loop instead of creating a new one
    future = asyncio.run_coroutine_threadsafe(_run(), _background_loop)
    return future.result()


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/api/health")
def api_health():
    return jsonify({"ok": True})


@app.post("/api/session")
def api_session_create():
    ensure_initialized()

    async def _create():
        assert _session_service is not None
        session = await _session_service.create_session(
            state={},
            app_name="adk_agent_samples",
            user_id="web_user",
        )
        return {"session_id": session.id, "user_id": session.user_id}

    # Use the background event loop
    future = asyncio.run_coroutine_threadsafe(_create(), _background_loop)
    result = future.result()
    return jsonify(result)


@app.post("/api/chat/send")
def api_chat_send():
    ensure_initialized()
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    text = data.get("text", "")
    user_id = data.get("user_id", "web_user")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    if not text:
        return jsonify({"error": "text required"}), 400

    try:
        messages = _collect_events_sync(session_id, user_id, text)
        return jsonify({"messages": messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------- Files (read-only) --------
@app.get("/api/files/calcdirs")
def api_files_calcdirs():
    tmp_root = ALLOWED_ROOTS["tmp"]
    os.makedirs(tmp_root, exist_ok=True)
    items = [
        name for name in os.listdir(tmp_root)
        if os.path.isdir(os.path.join(tmp_root, name))
    ]
    return jsonify({"calcdirs": sorted(items)})


@app.get("/api/files/list")
def api_files_list():
    # expects ?path=tmp/LaFeO3
    rel = request.args.get("path", "tmp")
    try:
        base = secure_path(rel)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.exists(base):
        return jsonify({"error": "path not found"}), 404

    result = []
    # 只遍历当前目录，不递归子目录
    try:
        entries = os.listdir(base)
        for entry in entries:
            entry_path = os.path.join(base, entry)
            if os.path.isdir(entry_path):
                result.append({
                    "type": "directory",
                    "path": os.path.relpath(entry_path, PROJECT_ROOT)
                })
            elif os.path.isfile(entry_path):
                result.append({
                    "type": "file",
                    "path": os.path.relpath(entry_path, PROJECT_ROOT)
                })
    except OSError as e:
        return jsonify({"error": f"Cannot list directory: {e}"}), 500
    return jsonify({"entries": result})


@app.get("/api/files/read")
def api_files_read():
    rel = request.args.get("path")
    if not rel:
        return jsonify({"error": "path required"}), 400
    try:
        p = secure_path(rel)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.exists(p):
        return jsonify({"error": "file not found"}), 404

    # text preview only
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return jsonify({"path": rel, "content": content})
    except Exception as e:
        return jsonify({"error": f"failed to read: {e}"}), 500


@app.get("/api/files/download")
def api_files_download():
    rel = request.args.get("path")
    if not rel:
        return jsonify({"error": "path required"}), 400
    try:
        p = secure_path(rel)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.exists(p):
        return jsonify({"error": "file not found"}), 404

    return send_file(p, as_attachment=True)

@app.get("/")

def index():
    # Serve the minimal single-page UI from static/index.html (now in project root)
    return send_file(os.path.join(PROJECT_ROOT, "static", "index.html"))


if __name__ == "__main__":
    print("🚀 VASP Agent 启动中...")
    print("📍 访问地址: http://localhost:5175")

    app.run(
        host="0.0.0.0",
        port=5175,
        debug=True,
        use_reloader=False,  # 关键：禁用文件监控重载
        threaded=True
    )

