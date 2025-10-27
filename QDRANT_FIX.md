# Qdrant Connection Fix

## 问题描述

Frontend启动时报错：
```
Qdrant seeding error: HTTPConnectionPool(host='localhost', port=6333):
Max retries exceeded with url: /collections/assessment_docs_minilm
(Caused by NewConnectionError: Failed to establish a new connection: [Errno 111] Connection refused)
```

## 根本原因

### 问题1：.env配置错误
- `.env`文件中设置了`QDRANT_HOST=localhost`
- 这个配置适合本地开发，但在Docker容器内部应该使用容器名称`qdrant`

### 问题2：Shell环境变量污染
- 之前的shell session中export了`QDRANT_HOST=localhost`和`QDRANT_SEED_PATH=ai-assessment-deploy/...`（相对路径）
- Docker Compose优先使用shell环境变量，覆盖了docker-compose.yml中的默认值

### 问题3：Seed文件路径错误
- `QDRANT_SEED_PATH`设置为相对路径，在容器内无法找到文件
- 应该使用绝对路径`/app/data/qdrant_seed/assessment_docs_minilm.jsonl`

## 解决方案

### 1. 修改.env文件
注释掉`QDRANT_HOST=localhost`，让Docker Compose使用默认值：

```bash
# Vector Database (Qdrant)
# Leave QDRANT_HOST unset for Docker (will use 'qdrant' container name)
# For local development, set: QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 2. 清理Shell环境变量
在运行docker-compose之前unset环境变量：

```bash
unset QDRANT_HOST QDRANT_SEED_PATH
docker-compose up -d
```

### 3. 使用提供的启动脚本（推荐）
使用`start.sh`脚本自动处理：

```bash
./start.sh
```

## 验证修复

### 1. 检查容器内环境变量
```bash
docker exec backend-api env | grep QDRANT_HOST
# 应该输出: QDRANT_HOST=qdrant
```

### 2. 测试Qdrant连接
```bash
docker exec backend-api curl -s http://qdrant:6333/collections/assessment_docs_minilm
# 应该返回collection信息的JSON
```

### 3. 检查容器状态
```bash
docker-compose ps
# 所有容器应该显示 (healthy)
```

### 4. 检查Backend日志
```bash
docker logs backend-api 2>&1 | grep "http://qdrant:6333"
# 应该看到成功的HTTP请求日志
```

## Docker vs 本地开发配置

### Docker部署（容器间通信）
- `QDRANT_HOST=qdrant` （容器名称）
- docker-compose.yml会自动配置DNS解析

### 本地开发（直接运行Python）
- `QDRANT_HOST=localhost`
- Qdrant需要在本机6333端口运行

## docker-compose.yml配置说明

```yaml
environment:
  - QDRANT_HOST=${QDRANT_HOST:-qdrant}  # 默认值qdrant
  - QDRANT_PORT=${QDRANT_PORT:-6333}
```

语法说明：
- `${VAR:-default}` 表示如果VAR未设置或为空，使用default
- 如果.env或shell中设置了QDRANT_HOST，会覆盖默认值
- 因此需要确保这些地方没有设置为localhost

## 最佳实践

1. **Docker部署**：不要在.env中设置QDRANT_HOST，使用默认值
2. **本地开发**：在本地shell中临时export：
   ```bash
   export QDRANT_HOST=localhost
   python -m backend.main
   ```
3. **使用start.sh**：统一启动方式，避免环境污染

## 相关文件

- `.env` - 环境变量配置
- `docker-compose.yml` - 容器编排配置
- `start.sh` - 启动脚本（自动unset）
- `backend/backend/config/settings.py` - 读取QDRANT_HOST配置
- `backend/backend/services/qdrant_seed.py` - Qdrant连接逻辑

## 修复确认

✅ QDRANT_HOST正确设置为`qdrant`
✅ Backend能成功连接Qdrant
✅ Collection `assessment_docs_minilm` 存在且有152,987个points
✅ 所有容器健康运行
✅ Frontend能正常获取seed status
