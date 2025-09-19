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
        instruction="帮助人类工作",
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

