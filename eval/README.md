# RAG Evaluation Results

## Task 3.2 - RAG System Evaluation

### 评估配置
- **评估时间：** 2025-10-28
- **Backend URL：** http://localhost:8888
- **测试集：** 60个问题（每类20个）
  - Metadata问题：20个
  - Keyword问题：20个
  - Semantic问题：20个
- **Random Seed：** 42（可复现）
- **Reranker：** fallback (MiniLM ONNX)
- **Vector Limit：** 10
- **Content Char Limit：** 500

### 整体性能指标

| 指标 | 值 |
|------|-----|
| **Top-5 准确率** | 73.3% (44/60) |
| **Top-1 准确率** | 45.0% (27/60) |
| **平均延迟** | 225.5ms |
| **中位数延迟** | 209.0ms |
| **P95延迟** | 345.1ms |

### 分类别性能

#### 1. Keyword Questions（关键词问题）
**表现最好** ✨
- Total: 20 questions
- **Top-5 Accuracy: 90.0%** (18/20)
- **Top-1 Accuracy: 60.0%** (12/20)
- Retrieval Time - Mean: 227.4ms, Median: 207.7ms
- Embed Time - Mean: 5.5ms, Median: 5.4ms
- Vector Time - Mean: 6.6ms, Median: 6.0ms
- Rerank Time - Mean: 215.0ms, Median: 196.3ms

#### 2. Semantic Questions（语义问题）
**表现良好** ✅
- Total: 20 questions
- **Top-5 Accuracy: 80.0%** (16/20)
- **Top-1 Accuracy: 40.0%** (8/20)
- Retrieval Time - Mean: 258.1ms, Median: 224.0ms
- Embed Time - Mean: 8.1ms, Median: 7.7ms
- Vector Time - Mean: 8.2ms, Median: 7.7ms
- Rerank Time - Mean: 241.6ms, Median: 207.7ms

#### 3. Metadata Questions（元数据问题）
**需要改进** ⚠️
- Total: 20 questions
- **Top-5 Accuracy: 50.0%** (10/20)
- **Top-1 Accuracy: 35.0%** (7/20)
- Retrieval Time - Mean: 191.0ms, Median: 182.4ms
- Embed Time - Mean: 4.8ms, Median: 4.9ms
- Vector Time - Mean: 5.6ms, Median: 5.6ms
- Rerank Time - Mean: 180.3ms, Median: 172.3ms

### 延迟分析

#### 各阶段耗时占比
1. **Reranking** - 占主要时间（~200ms，约89%）
2. **Vector Search** - 很快（~6ms，约3%）
3. **Embedding** - 很快（~6ms，约3%）
4. **其他开销** - LLM生成等（~13ms，约5%）

#### 性能优化建议
- ✅ Embedding已优化（ONNX INT8量化）
- ✅ Vector search已优化（Qdrant HNSW索引）
- ⚠️ Reranking是主要瓶颈，可以考虑：
  - 使用更快的reranker模型
  - 减少rerank的候选数量
  - 考虑GPU加速

### 文件说明

- **eval_RAG_task3_2.json** - 完整的评估结果（JSON格式）
  - 包含每个问题的详细结果
  - 包含timing信息
  - 包含准确率统计

- **eval_RAG_task3_2.txt** - 评估过程输出（文本格式）
  - 包含每个问题的实时评估过程
  - 包含最终总结报告
  - 便于阅读和分享

### 如何重新运行评估

```bash
cd /Users/yilu/Downloads/ai_assessment_project/ai-assessment-deploy

# 运行评估并保存结果
python3 -u scripts/eval_rag_20x20x20.py \
  --backend http://localhost:8888 \
  --output eval/eval_RAG_task3_2.json \
  > eval/eval_RAG_task3_2.txt 2>&1 &

# 监控进度
tail -f eval/eval_RAG_task3_2.txt

# 或使用不同的seed
python3 -u scripts/eval_rag_20x20x20.py \
  --backend http://localhost:8888 \
  --seed 123 \
  --output eval/eval_RAG_task3_2_seed123.json \
  > eval/eval_RAG_task3_2_seed123.txt 2>&1
```

### 修复历史

**问题：** 最初所有评估结果显示0%准确率

**根因：** API返回`citations`字段，但评估脚本期望`chunks`字段

**修复：** 修改 `eval_rag_20x20x20.py` Line 371
```python
# 修改前
chunks = data.get("chunks", [])

# 修改后
chunks = data.get("citations", [])  # API returns "citations", not "chunks"
```

### 结论

RAG系统整体表现良好：
- ✅ Keyword检索非常准确（90% Top-5）
- ✅ Semantic检索表现良好（80% Top-5）
- ⚠️ Metadata检索需要改进（50% Top-5）
- ✅ 延迟控制在可接受范围（平均225ms）
- ⚠️ Reranking是主要性能瓶颈

**推荐改进方向：**
1. 优化metadata字段的索引和检索策略
2. 考虑使用更快的reranker或GPU加速
3. 调整vector_limit和content_char_limit参数平衡准确率和性能
