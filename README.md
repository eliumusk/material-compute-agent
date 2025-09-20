# 🧪 VASP Agent - Intelligent Materials Computation Assistant

[中文版本](#中文版本) | [English Version](#english-version)

---

## English Version

### 🌟 Overview

VASP Agent is an intelligent materials computation assistant that integrates literature analysis, materials modeling, VASP configuration, task execution, result analysis, and report generation. It provides both a command-line interface and a modern web interface for comprehensive materials science workflows.

### ✨ Key Features

- **📚 Literature Analysis**: Extract material information from PDF documents
- **🔬 Structure Generation**: Generate and optimize POSCAR structure files
- **⚙️ VASP Configuration**: Automated INCAR, KPOINTS, and POTCAR setup
- **☁️ Cloud Computing**: Submit and monitor calculations on Bohrium platform
- **📊 Result Analysis**: Analyze vasprun.xml and generate comprehensive reports
- **🌐 Multi-interface**: Both web UI and command-line interfaces
- **🌍 Internationalization**: Support for Chinese and English languages

### 🚀 Quick Start

#### Prerequisites

- Python 3.11+
- Conda environment
- Bohrium account with valid credentials
- VASP license (for calculations)

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dptech-corp/material-compute-agent
   cd material-compute-agent
   ```

2. **Set up environment**
   ```bash
   conda create -n bohr-agent python=3.11
   conda activate bohr-agent
   pip install -e .
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade google-adk
   git clone -b lh https://github.com/dptech-corp/science-agent-sdk.git
   pip install -e ./science-agent-sdk
   ```

4. **Configure credentials**
   Create a `.env` file in the project root:
   ```env
   BOHRIUM_USERNAME=your_email@example.com
   BOHRIUM_PASSWORD=your_password
   BOHRIUM_PROJECT_ID=your_project_id
   OPENROUTER_API_KEY=your_openrouter_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

#### Usage Options

**Option 1: Web Interface (Recommended)**
```bash
# Start MCP server (in one terminal)
cd server
python CalculationMCPServer.py

# Start web application (in another terminal)
python app.py
```
Then open http://localhost:5175 in your browser.

**Option 2: Command Line Interface**
```bash
# Start MCP server first
cd server
python CalculationMCPServer.py

# Run agent in another terminal
python agent.py
```

### 🖥️ Web Interface Guide

#### Main Features

1. **💬 Chat Interface**
   - Interactive conversation with the AI assistant
   - Support for both Chinese and English
   - Real-time message streaming
   - Markdown rendering for formatted responses

2. **📁 File Management**
   - Browse calculation directories
   - Preview VASP input/output files
   - Download files directly from the interface

3. **📊 Results Analysis**
   - View calculation results organized by timestamp
   - Quick access to vasprun.xml files
   - Integrated analysis tools

4. **📖 Literature Management**
   - Upload and manage PDF documents
   - Automatic literature analysis
   - Extract material information for calculations

5. **📝 Logs Monitoring**
   - Real-time calculation logs
   - Error tracking and debugging
   - Task status monitoring

#### Language Switching

The web interface supports both Chinese and English. Use the language selector in the chat header to switch between languages. Your preference is automatically saved.

### 🔧 Configuration

#### VASP Settings

The agent automatically configures VASP parameters based on:
- Material composition
- Calculation type (relaxation, static, etc.)
- Literature recommendations
- Best practices for the material system

#### Bohrium Integration

- **Platform**: Alibaba Cloud computing resources
- **Image**: `registry.dp.tech/dptech/vasp:5.4.4`
- **Resources**: Configurable CPU/memory allocation
- **Storage**: Automatic file synchronization

### 📋 Workflow Example

1. **Start a new session**
   ```
   User: "I want to calculate the properties of Sr5Ca3Fe8O24"
   ```

2. **Literature analysis** (optional)
   ```
   User: "Please analyze this PDF: test.pdf"
   ```

3. **Structure generation**
   - Agent searches for POSCAR templates
   - Performs atomic substitution
   - Validates structure composition

4. **VASP configuration**
   - Generates INCAR, KPOINTS, POTCAR
   - Optimizes parameters for the material system
   - Requests user confirmation

5. **Task submission**
   - Submits to Bohrium platform
   - Monitors calculation progress
   - Downloads results automatically

6. **Result analysis**
   - Parses vasprun.xml
   - Generates comprehensive report
   - Provides insights and recommendations

### 🛠️ Advanced Features

#### Custom VASP Parameters

You can modify VASP parameters during the workflow:
```
User: "Set NELM = 100 in INCAR"
Agent: "I'll update the INCAR file with NELM = 100..."
```

#### Batch Calculations

Submit multiple related calculations:
```
User: "Calculate properties for both Sr5Ca3Fe8O24 and Sr4Ca4Fe8O24"
```

#### Result Comparison

Compare results across different calculations:
```
User: "Compare the band gaps between these two structures"
```

### 🔍 Troubleshooting

#### Common Issues

1. **MCP Connection Failed**
   - Ensure MCP server is running on port 8000
   - Check firewall settings
   - Verify network connectivity

2. **Bohrium Authentication Error**
   - Verify credentials in `.env` file
   - Check project ID and permissions
   - Ensure account has sufficient resources

3. **VASP Calculation Errors**
   - Check POSCAR structure validity
   - Verify POTCAR availability
   - Review INCAR parameters

#### Debug Mode

Enable debug logging:
```bash
export DEBUG=1
python app.py
```

### 📚 API Reference

#### REST Endpoints

- `POST /api/chat/send` - Send message to agent
- `GET /api/files/list` - List files in directory
- `GET /api/files/read` - Read file content
- `GET /api/session/new` - Create new session

#### MCP Tools

- `vasp_job` - Submit VASP calculation
- `analyze_vasprun_all` - Analyze calculation results
- `write_poscar` - Generate structure files
- `write_vasp_config` - Configure VASP parameters

### 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- [Bohrium Platform](https://bohrium.dp.tech/) for cloud computing resources
- [VASP](https://www.vasp.at/) for the calculation engine
- [Google ADK](https://github.com/google/adk) for the agent framework

---

## 中文版本

### 🌟 概述

VASP Agent 是一个集文献分析、材料建模、VASP 配置、任务执行、结果分析和报告生成于一体的智能材料计算助手。提供命令行和现代化 Web 界面，支持完整的材料科学计算工作流。

### ✨ 主要功能

- **📚 文献分析**: 从 PDF 文档中提取材料信息
- **🔬 结构生成**: 生成和优化 POSCAR 结构文件
- **⚙️ VASP 配置**: 自动化 INCAR、KPOINTS 和 POTCAR 设置
- **☁️ 云端计算**: 在 Bohrium 平台提交和监控计算任务
- **📊 结果分析**: 分析 vasprun.xml 并生成综合报告
- **🌐 多界面**: 支持 Web UI 和命令行界面
- **🌍 国际化**: 支持中英文双语

### 🚀 快速开始

#### 环境要求

- Python 3.11+
- Conda 环境
- Bohrium 账户和有效凭据
- VASP 许可证（用于计算）

#### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/dptech-corp/material-compute-agent
   cd material-compute-agent
   ```

2. **设置环境**
   ```bash
   conda create -n bohr-agent python=3.11
   conda activate bohr-agent
   pip install -e .
   ```

3. **安装依赖**
   ```bash
   pip install --upgrade google-adk
   git clone -b lh https://github.com/dptech-corp/science-agent-sdk.git
   pip install -e ./science-agent-sdk
   ```

4. **配置凭据**
   在项目根目录创建 `.env` 文件：
   ```env
   BOHRIUM_USERNAME=your_email@example.com
   BOHRIUM_PASSWORD=your_password
   BOHRIUM_PROJECT_ID=your_project_id
   OPENROUTER_API_KEY=your_openrouter_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

#### 使用方式

**方式一：Web 界面（推荐）**
```bash
# 启动 MCP 服务器（一个终端）
cd server
python CalculationMCPServer.py

# 启动 Web 应用（另一个终端）
python app.py
```
然后在浏览器中打开 http://localhost:5175

**方式二：命令行界面**
```bash
# 先启动 MCP 服务器
cd server
python CalculationMCPServer.py

# 在另一个终端运行 agent
python agent.py
```

### 🖥️ Web 界面指南

#### 主要功能

1. **💬 对话界面**
   - 与 AI 助手进行交互式对话
   - 支持中英文双语
   - 实时消息流
   - Markdown 格式渲染

2. **📁 文件管理**
   - 浏览计算目录
   - 预览 VASP 输入/输出文件
   - 直接从界面下载文件

3. **📊 结果分析**
   - 按时间戳查看计算结果
   - 快速访问 vasprun.xml 文件
   - 集成分析工具

4. **📖 文献管理**
   - 上传和管理 PDF 文档
   - 自动文献分析
   - 提取材料信息用于计算

5. **📝 日志监控**
   - 实时计算日志
   - 错误跟踪和调试
   - 任务状态监控

#### 语言切换

Web 界面支持中英文双语。使用对话标题栏中的语言选择器切换语言，您的偏好会自动保存。

### 🔧 配置说明

#### VASP 设置

Agent 会根据以下因素自动配置 VASP 参数：
- 材料组成
- 计算类型（结构优化、静态计算等）
- 文献建议
- 材料体系的最佳实践

#### Bohrium 集成

- **平台**: 玻尔计算资源
- **镜像**: `registry.dp.tech/dptech/vasp:5.4.4`
- **资源**: 可配置的 CPU/内存分配
- **存储**: 自动文件同步

### 📋 工作流示例

1. **开始新会话**
   ```
   用户: "我想计算 Sr5Ca3Fe8O24 的性质"
   ```

2. **文献分析**（可选）
   ```
   用户: "请分析这个 PDF: test.pdf"
   ```

3. **结构生成**
   - Agent 搜索 POSCAR 模板
   - 执行原子替换
   - 验证结构组成

4. **VASP 配置**
   - 生成 INCAR、KPOINTS、POTCAR
   - 为材料体系优化参数
   - 请求用户确认

5. **任务提交**
   - 提交到 Bohrium 平台
   - 监控计算进度
   - 自动下载结果

6. **结果分析**
   - 解析 vasprun.xml
   - 生成综合报告
   - 提供见解和建议

### 🛠️ 高级功能

#### 自定义 VASP 参数

您可以在工作流中修改 VASP 参数：
```
用户: "在 INCAR 中设置 NELM = 100"
Agent: "我将更新 INCAR 文件，设置 NELM = 100..."
```

#### 批量计算

提交多个相关计算：
```
用户: "计算 Sr5Ca3Fe8O24 和 Sr4Ca4Fe8O24 的性质"
```

#### 结果比较

比较不同计算的结果：
```
用户: "比较这两个结构的带隙"
```

### 🔍 故障排除

#### 常见问题

1. **MCP 连接失败**
   - 确保 MCP 服务器在端口 8000 上运行
   - 检查防火墙设置
   - 验证网络连接

2. **Bohrium 认证错误**
   - 验证 `.env` 文件中的凭据
   - 检查项目 ID 和权限
   - 确保账户有足够的资源

3. **VASP 计算错误**
   - 检查 POSCAR 结构有效性
   - 验证 POTCAR 可用性
   - 审查 INCAR 参数

#### 调试模式

启用调试日志：
```bash
export DEBUG=1
python app.py
```

### 📚 API 参考

#### REST 端点

- `POST /api/chat/send` - 向 agent 发送消息
- `GET /api/files/list` - 列出目录中的文件
- `GET /api/files/read` - 读取文件内容
- `GET /api/session/new` - 创建新会话

#### MCP 工具

- `vasp_job` - 提交 VASP 计算
- `analyze_vasprun_all` - 分析计算结果
- `write_poscar` - 生成结构文件
- `write_vasp_config` - 配置 VASP 参数

### 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

### 🙏 致谢

- [Bohrium 平台](https://bohrium.dp.tech/) 提供云计算资源
- [VASP](https://www.vasp.at/) 提供计算引擎
- [Google ADK](https://github.com/google/adk) 提供 agent 框架
