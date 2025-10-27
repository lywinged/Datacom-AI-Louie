#!/usr/bin/env python3
"""
Translate Chinese comments and strings in app.py to English
"""

translations = {
    # Docstrings
    "管理会话状态和消息历史的 SQLite 数据库": "SQLite database for managing session state and message history",
    "确保数据库表存在": "Ensure database tables exist",
    "从数据库加载约束": "Load constraints from database",
    "保存约束到数据库": "Save constraints to database",
    "添加消息到历史": "Add message to history",
    "加载消息历史": "Load message history",

    # Comments - Function headers
    "约束提取函数": "Constraint extraction functions",
    "从用户输入中提取约束信息（正则方式）": "Extract constraint information from user input (regex approach)",
    "提取预算": "Extract budget",
    "提取天数": "Extract number of days",
    "提取目的地和出发地": "Extract destination and departure",
    "通用 to/from 模式": "Generic to/from pattern",
    "放在工具函数区": "Place in utility functions section",
    "判断文本中是否可能包含未提取的城市/地点信息": "Determine if text may contain unextracted city/location information",
    "使用 LLM 提取约束信息（兜底方案）": "Use LLM to extract constraint information (fallback approach)",
    "混合提取：先用正则，提取不到关键信息时用 LLM 兜底": "Hybrid extraction: try regex first, fallback to LLM if key information not found",
    "检查四要素是否完整": "Check if all four required elements are complete",
    "验证约束是否合理": "Validate if constraints are reasonable",
    "格式化约束摘要": "Format constraint summary",

    # UI section comments
    "Streamlit UI 配置": "Streamlit UI Configuration",
    "初始化 session state": "Initialize session state",
    "指标跟踪": "Metrics tracking",
    "抢先消费 pending_prompt，避免被首启提示/按钮再次打断": "Consume pending_prompt early to avoid interruption by startup prompt/buttons",
    "有 pending 时，跳过首启引导，直接进入对应分支": "Skip startup guide when pending exists, go directly to corresponding branch",
    "不跳过模式激活时的首启": "Don't skip startup when mode is activated",

    # Sidebar comments
    "侧边栏 - 服务状态和评估仪表板": "Sidebar - Service Status and Evaluation Dashboard",
    "检查服务状态": "Check service status",
    "显示服务状态": "Display service status",
    "基础统计": "Basic statistics",
    "Latency & Cost 趋势 - 始终显示": "Latency & Cost Trends - Always display",
    "显示平均值和中位数": "Display mean and median",
    "Cost 趋势 - 始终显示": "Cost Trends - Always display",
    "RAG 性能指标": "RAG Performance Metrics",
    "检索时间统计": "Retrieval time statistics",
    "准确率/置信度趋势": "Accuracy/Confidence trends",
    "Agent 成功率 - 始终显示": "Agent Success Rate - Always display",

    # Main interface
    "主界面": "Main Interface",
    "快捷按钮（3个：RAG, Trip, Code）": "Quick action buttons (3: RAG, Trip, Code)",
    "显示聊天历史": "Display chat history",
    "聊天输入": "Chat input",
    "如果 early_prompt 已经提供了 prompt，直接使用": "If early_prompt already provided prompt, use it directly",
    "否则检查 pending_prompt": "Otherwise check pending_prompt",
    "最后尝试使用用户输入": "Finally try user input",
    "显示用户消息": "Display user message",

    # Mode-specific comments
    "RAG Mode - 完整复刻 chat_rag.py": "RAG Mode - Full replication of chat_rag.py",
    "Trip Planning Mode - 完整复刻 chat_agent.py": "Trip Planning Mode - Full replication of chat_agent.py",
    "显示答案": "Display answer",
    "显示指标": "Display metrics",
    "显示引用来源": "Display citation sources",
    "询问是否继续": "Ask if continue",
    "收集RAG指标": "Collect RAG metrics",
    "保存到消息历史并立即显示": "Save to message history and display immediately",
    "特殊命令：status": "Special command: status",
    "如果正在等待确认": "If waiting for confirmation",
    "执行规划": "Execute planning",
    "填充开始日期": "Fill start date",
    "费用": "Cost",
    "约束满足情况": "Constraint satisfaction status",
    "正常约束收集流程": "Normal constraint collection flow",
    "思考步骤：提取约束": "Thinking step: extract constraints",
    "检查是否有新信息": "Check if there is new information",
    "显示提取的信息": "Display extracted information",
    "检查完整性": "Check completeness",
    "验证合理性": "Validate reasonableness",
    "信息完整且合理，询问确认": "Information complete and reasonable, ask for confirmation",
    "检测语言": "Detect language",
    "检查并安装必要的工具链": "Check and install necessary toolchain",
    "伪流式进度显示": "Pseudo-streaming progress display",
    "开始进度动画": "Start progress animation",
    "记录开始时间": "Record start time",
    "实际 API 调用": "Actual API call",
    "计算延迟": "Calculate latency",
    "显示状态": "Display status",
    "显示元数据": "Display metadata",
    "显示生成的代码": "Display generated code",
    "收集Code指标到dashboard": "Collect Code metrics to dashboard",

    # Intent recognition
    "自动意图识别（无 mode 时）- 使用 LLM 分析意图": "Automatic intent recognition (when no mode) - Use LLM to analyze intent",
    "本地兜底：不给 OpenAI 也能回": "Local fallback: can respond even without OpenAI",
    "使用LLM分析意图": "Use LLM to analyze intent",
    "根据意图路由": "Route based on intent",
    "General Assistant 模式 - 直接用LLM回答": "General Assistant mode - Answer directly with LLM",
    "构建对话历史（最近5条）": "Build conversation history (last 5 messages)",
    "限制长度，确保字符": "Limit length, ensure characters",
    "添加系统提示": "Add system prompt",
    "显示回复": "Display response",
    "显示服务提示": "Display service hint",
    "保存到历史": "Save to history",

    # User-facing strings
    "嗨！我已经收到你的消息啦 👋\n\n目前没有配置 OPENAI_API_KEY，所以先用本地兜底回复。\n你可以继续问我：\n- 输入 "trip …" 让我进入行程规划\n- 输入 "rag …" 让我查文档\n- 输入 "code …" 让我写代码\n":
        "Hi! I've received your message 👋\n\nOPENAI_API_KEY is not configured, so I'm using a local fallback response.\nYou can continue by:\n- Type \"trip ...\" for trip planning\n- Type \"rag ...\" for document search\n- Type \"code ...\" for code generation\n",

    "👋 已退出 RAG 模式。": "👋 Exited RAG mode.",
}

def translate_file(input_path, output_path):
    """Translate Chinese text in the file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply all translations
    for chinese, english in translations.items():
        content = content.replace(chinese, english)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Translation complete: {output_path}")
    print(f"   Applied {len(translations)} translations")

if __name__ == "__main__":
    input_file = "/Users/yilu/Downloads/ai_assessment_project/ai-assessment-deploy/frontend/app.py"
    output_file = "/Users/yilu/Downloads/ai_assessment_project/ai-assessment-deploy/frontend/app.py"

    # Backup original file
    import shutil
    backup_file = input_file + ".backup_chinese"
    shutil.copy(input_file, backup_file)
    print(f"📁 Backup created: {backup_file}")

    translate_file(input_file, output_file)
