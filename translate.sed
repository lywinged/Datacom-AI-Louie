# Translate Chinese docstrings and comments to English

# Docstrings
s/管理会话状态和消息历史的 SQLite 数据库/SQLite database for managing session state and message history/g
s/确保数据库表存在/Ensure database tables exist/g
s/从数据库加载约束/Load constraints from database/g
s/保存约束到数据库/Save constraints to database/g
s/添加消息到历史/Add message to history/g
s/加载消息历史/Load message history/g

# Comments - Function sections
s/约束提取函数/Constraint extraction functions/g
s/从用户输入中提取约束信息（正则方式）/Extract constraint information from user input (regex approach)/g
s/提取预算/Extract budget/g
s/提取天数/Extract number of days/g
s/提取目的地和出发地/Extract destination and departure/g
s/通用 to\/from 模式/Generic to\/from pattern/g
s/放在工具函数区/Place in utility functions section/g
s/判断文本中是否可能包含未提取的城市\/地点信息/Determine if text may contain unextracted city\/location information/g
s/使用 LLM 提取约束信息（兜底方案）/Use LLM to extract constraint information (fallback approach)/g
s/混合提取：先用正则，提取不到关键信息时用 LLM 兜底/Hybrid extraction: try regex first, fallback to LLM if key information not found/g
s/检查四要素是否完整/Check if all four required elements are complete/g
s/验证约束是否合理/Validate if constraints are reasonable/g
s/格式化约束摘要/Format constraint summary/g

# UI section comments
s/Streamlit UI 配置/Streamlit UI Configuration/g
s/初始化 session state/Initialize session state/g
s/指标跟踪/Metrics tracking/g
s/抢先消费 pending_prompt，避免被首启提示\/按钮再次打断/Consume pending_prompt early to avoid interruption by startup prompt\/buttons/g
s/有 pending 时，跳过首启引导，直接进入对应分支/Skip startup guide when pending exists, go directly to corresponding branch/g
s/不跳过模式激活时的首启/Don't skip startup when mode is activated/g

# Sidebar comments
s/侧边栏 - 服务状态和评估仪表板/Sidebar - Service Status and Evaluation Dashboard/g
s/检查服务状态/Check service status/g
s/显示服务状态/Display service status/g
s/基础统计/Basic statistics/g
s/Latency & Cost 趋势 - 始终显示/Latency & Cost Trends - Always display/g
s/显示平均值和中位数/Display mean and median/g
s/Cost 趋势 - 始终显示/Cost Trends - Always display/g
s/RAG 性能指标/RAG Performance Metrics/g
s/检索时间统计/Retrieval time statistics/g
s/准确率\/置信度趋势/Accuracy\/Confidence trends/g
s/Agent 成功率 - 始终显示/Agent Success Rate - Always display/g

# Main interface
s/主界面/Main Interface/g
s/快捷按钮（3个：RAG, Trip, Code）/Quick action buttons (3: RAG, Trip, Code)/g
s/显示聊天历史/Display chat history/g
s/聊天输入/Chat input/g
s/如果 early_prompt 已经提供了 prompt，直接使用/If early_prompt already provided prompt, use it directly/g
s/否则检查 pending_prompt/Otherwise check pending_prompt/g
s/最后尝试使用用户输入/Finally try user input/g
s/显示用户消息/Display user message/g

# Mode-specific
s/RAG Mode - 完整复刻 chat_rag.py/RAG Mode - Full replication of chat_rag.py/g
s/Trip Planning Mode - 完整复刻 chat_agent.py/Trip Planning Mode - Full replication of chat_agent.py/g
s/显示答案/Display answer/g
s/显示指标/Display metrics/g
s/显示引用来源/Display citation sources/g
s/询问是否继续/Ask if continue/g
s/收集RAG指标/Collect RAG metrics/g
s/保存到消息历史并立即显示/Save to message history and display immediately/g
s/特殊命令：status/Special command: status/g
s/如果正在等待确认/If waiting for confirmation/g
s/执行规划/Execute planning/g
s/填充开始日期/Fill start date/g
s/费用/Cost/g
s/约束满足情况/Constraint satisfaction status/g
s/正常约束收集流程/Normal constraint collection flow/g
s/思考步骤：提取约束/Thinking step: extract constraints/g
s/检查是否有新信息/Check if there is new information/g
s/显示提取的信息/Display extracted information/g
s/检查完整性/Check completeness/g
s/验证合理性/Validate reasonableness/g
s/信息完整且合理，询问确认/Information complete and reasonable, ask for confirmation/g
s/检测语言/Detect language/g
s/检查并安装必要的工具链/Check and install necessary toolchain/g
s/伪流式进度显示/Pseudo-streaming progress display/g
s/开始进度动画/Start progress animation/g
s/记录开始时间/Record start time/g
s/实际 API 调用/Actual API call/g
s/计算延迟/Calculate latency/g
s/显示状态/Display status/g
s/显示元数据/Display metadata/g
s/显示生成的代码/Display generated code/g
s/收集Code指标到dashboard/Collect Code metrics to dashboard/g

# Intent recognition
s/自动意图识别（无 mode 时）- 使用 LLM 分析意图/Automatic intent recognition (when no mode) - Use LLM to analyze intent/g
s/本地兜底：不给 OpenAI 也能回/Local fallback: can respond even without OpenAI/g
s/使用LLM分析意图/Use LLM to analyze intent/g
s/根据意图路由/Route based on intent/g
s/General Assistant 模式 - 直接用LLM回答/General Assistant mode - Answer directly with LLM/g
s/构建对话历史（最近5条）/Build conversation history (last 5 messages)/g
s/限制长度，确保字符/Limit length, ensure characters/g
s/添加系统提示/Add system prompt/g
s/显示回复/Display response/g
s/显示服务提示/Display service hint/g
s/保存到历史/Save to history/g

# User-facing messages
s/👋 已退出 RAG 模式。/👋 Exited RAG mode./g
