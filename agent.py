import os
import nest_asyncio
import asyncio
nest_asyncio.apply()
import subprocess
from google.adk.tools.mcp_tool.mcp_toolset import SseConnectionParams
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dp.agent.adapter.adk import CalculationMCPToolset
from google.adk.tools import FunctionTool
from vasp_function import read_vasp_pdf,write_vasp_report,analyze_vasprun_all,search_poscar_template,write_poscar,write_vasp_config
from vasp_function import show_vasp_config,rewrite_vasp_config
from utils import ask_human_for_advice,show_task_status
from dotenv import load_dotenv
load_dotenv()


# 更新PATH环境变量（记得展开~成绝对路径）
home_dir = os.path.expanduser("~")
bohr_path = os.path.join(home_dir, ".bohrium")
os.environ["PATH"] += f":{bohr_path}"


os.environ["MP_ROOT_DIR"] = "mp_materials_cif" #mp_materials_cif material project


current_dir = os.getcwd()
bohr_executor = {
    "type": "dispatcher",
    "machine": {
        "batch_type": "Bohrium",
        "context_type": "Bohrium",
        "remote_profile": {
            "email": "linhang@dp.tech",
            "password": "DP123456",
            "project_id": 21128,
            "program_id": 21128,
            "input_data": {
                "image_name": "registry.dp.tech/dptech/vasp:5.4.4",
                "job_type": "container",
                "platform": "ali",
                "scass_type": "c32_m32_cpu",
            },
        },
    },
    "DEFAULT_FORWARD_DIR":[f"{current_dir}/tmp"]
    
}
bohr_storage = {
    "type": "bohrium",
    "username": "linhang@dp.tech",
    "password": "DP123456",
    "project_id": 21128,
}



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

## ⚠️ 交互规则（必须遵守）：

- 每一个阶段都使用**ReAct模式**：先说明你要做什么（Reason），再执行（Action）
- 运行每一步之后，输出结果并使用`ask_human_for_advice`**请求用户确认**是否继续。
- 若中间出错，需立即停止并向用户反馈问题原因。
- **所有参数必须来自文献或用户确认**。
- 所有工具（如 `read_vasp_pdf`, `write_poscar`, `analyze_vasprun_all` 等）都作为你可以调用的功能模块。
- 你是一个严谨、高效、清晰的科研助手，禁止跳步或模糊执行。

---

当你准备好执行任务，使用ask_human_for_advice 向用户询问论文路径和目标化学式：
我已理解任务，将逐步引导您完成一次完整的 VASP 计算流程。
请提供论文路径或目标化学式：


'''

read_vasp_pdf = FunctionTool(func=read_vasp_pdf)
write_vasp_report = FunctionTool(func=write_vasp_report)
analyze_vasprun_all = FunctionTool(func=analyze_vasprun_all)
search_poscar_template = FunctionTool(func=search_poscar_template)
write_poscar = FunctionTool(func=write_poscar)
write_vasp_config = FunctionTool(func=write_vasp_config)
show_vasp_config = FunctionTool(func=show_vasp_config)
rewrite_vasp_config = FunctionTool(func=rewrite_vasp_config)
ask_human_for_advice = FunctionTool(func=ask_human_for_advice)
show_task_status = FunctionTool(func=show_task_status)


async def async_main():
    vasp_tools = await CalculationMCPToolset(
        connection_params=SseConnectionParams(
            url="http://localhost:8000/sse",
        ),
        executor=bohr_executor,
        storage=bohr_storage,
    ).get_tools()

    vasp_agent = LlmAgent(
        model=LiteLlm(model="openrouter/openai/gpt-4o"),
        name="vasp_agent",
        description=(
            "A phd who is good at using VASP to calculate the properties of materials."
        ),
        instruction="帮助人类工作",
        tools=[show_task_status,ask_human_for_advice,read_vasp_pdf,write_poscar,write_vasp_config,show_vasp_config,rewrite_vasp_config,analyze_vasprun_all,write_vasp_report,search_poscar_template,*vasp_tools],
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        state={},
        app_name="adk_agent_samples",
        user_id="vasp_agent"
    )

    runner = Runner(
        app_name="adk_agent_samples",
        agent=vasp_agent,
        session_service=session_service,
    )

    print("✅ Vasp任务已启动，输入 'exit' 结束对话。\n")

    while True:
        user_input = input("🧑 请输入指令：")  #
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Vasp任务已结束。")
            break

        content = types.Content(role="user", parts=[types.Part(text=user_input)])
        events_async = runner.run_async(
            session_id=session.id,
            user_id=session.user_id,
            new_message=content
        )

        async for event in events_async:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        role = event.content.role
                        if role == "user":
                            print(f"🧑 用户：{part.text}")
                        elif role == "model":
                            print(f"🤖 智能体：{part.text}")

if __name__ == "__main__":
    asyncio.run(async_main())



