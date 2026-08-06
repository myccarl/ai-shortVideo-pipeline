# myAiVideos

> 自动化短视频生产管线 —— 一条飞书指令，生成可发布的解说类成品视频。

[English](README.md) | **简体中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 简介

`myAiVideos` 是一套面向中文短视频内容生产的开源工程，覆盖**选题 → 创意 → 视觉 → 音频 → 后期 → 分发**全链路。以 FastAPI 为编排核心，Java 网关作中台治理层（鉴权 / 路由 / 熔断 / 计量 / 可观测），调度多种 AI 模型完成生成与质量门控。

适合的内容形态：

- 热搜解说 / 新闻快评
- 知识科普
- 情感故事
- 奇闻趣事
- 社会观察

## 核心特性

- **七层管线架构**：选题（L1）→ 创意（L2）→ 视觉（L3）→ 音频（L4）→ 后期（L5）→ 分发（L6）→ 优化（L7），层间解耦、可独立替换
- **多模型 failover + 熔断**：Java 网关聚合 DeepSeek / Qwen / GLM，Resilience4j 熔断保护，模型故障自动切换
- **AI 一致性治理**
  - Prompt 锚定：多段视频主体视觉一致性
  - CLIP 图文一致性门控：拦截跑题关键帧
  - 音画同步自动救援：4 档策略（音频变速 / 视频补齐 / 旁白重写）
- **全链路可观测**：trace_id 跨 Java/Python 双语言透传，Langfuse 调用树观测
- **计量与限流**：AOP 切面无侵入统计 token / 成本，按租户聚合
- **生产化能力**：SSE 实时进度、单段重生成、滚动日志归档

## 技术栈

| 层 | 组件 |
|---|---|
| 编排（Python） | FastAPI · asyncio · Pydantic · SQLAlchemy(asyncpg) · Alembic · ARQ |
| 网关（Java） | Spring Boot 3.5 · WebClient · Resilience4j · Caffeine · Prometheus |
| 前端 | Vue 3 · Vite · Vue Flow · Pinia · Tailwind |
| 数据 | PostgreSQL · Redis · MinIO |
| 可观测 | Langfuse · trace context propagation |
| AI 模型 | DeepSeek · Zhipu GLM-4V · Kling v2.5 · 火山引擎 / MiniMax TTS · faster-whisper |
| 容器 | Docker Compose |

## 架构总览

```
                +----------------+
                |   前端 / 飞书  |
                +--------+-------+
                         |
                +--------v-------+
                |  Java Gateway  |  JWT / TraceId / Router / Breaker / Meter
                +--------+-------+
                         |
                +--------v-------+
                |  FastAPI 编排  |  webhook · session · 任务编排
                +--+---------+---+
                   |         |
        +----------v---+   +-v-----------+
        |  ARQ Worker  |   | 7 层流水线  |
        +-----+--------+   +-+-----------+
              |              |
   +----------v--+    +------v-----------------+
   | Redis(队列) |    | DeepSeek / Kling / TTS |
   +-------------+    +------------------------+
```

更多细节见 [`architecture.md`](architecture.md)。

## 快速开始

### 环境要求

- Docker Desktop ≥ 24
- Node.js ≥ 20（仅前端开发用）
- Python ≥ 3.11（仅本地直跑 / 跑测试用）

### 一键启动

```bash
# 1. 克隆并准备环境变量
git clone <your-fork>.git
cd myAiVideos
cp .env.example .env
# 编辑 .env，至少填入 AI 模型 API key

# 2. 启动后端（postgres + redis + orchestrator + worker）
docker compose up -d postgres redis orchestrator worker

# 3. 跑数据库迁移
docker compose exec -T orchestrator alembic upgrade head

# 4.（可选）启动 Java 网关
docker compose up -d gateway

# 5. 启动前端
cd sl-vue && npm install && npm run dev
```

健康检查：

- Python API：<http://localhost:8000/health>
- Java Gateway：<http://localhost:8080/actuator/health>
- 前端：<http://localhost:5173>

详细步骤、不同场景启动方式、故障排查见 [`docs/启动-停止脚本.md`](docs/启动-停止脚本.md)。

### 配置说明

核心环境变量（完整列表见 `.env.example`）：

| 变量 | 用途 |
|---|---|
| `DEEPSEEK_API_KEY` | 文本 LLM |
| `GLM_API_KEY` | 多模态审核 |
| `KLING_ACCESS_KEY` / `KLING_SECRET_KEY` | 文生图 / 图生视频 |
| `ATLASCLOUD_API_KEY` | 可选 Atlas Cloud Media API 后端；当 `VISUAL_IMAGE_PROVIDER` 或 `VISUAL_VIDEO_PROVIDER` 设为 `atlascloud` 时使用 |
| `VOLCENGINE_TTS_*` 或 `MINIMAX_TTS_*` | 语音合成 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 链路观测（可选）|
| `GATEWAY_AUTH_JWT_SECRET` | Java 网关 JWT 签名（≥ 32 字节）|

## 项目结构

```
myAiVideos/
├── api/                  FastAPI 路由（webhook、storyboard、clip）
├── config/               配置、Skill 模板、角色、风格
├── core/                 trace context、langfuse 客户端、日志、调度
├── db/                   SQLAlchemy 模型、Alembic 迁移
├── integrations/         外部模型客户端封装
├── layers/               七层管线主体
│   ├── L1_trending/      选题：热搜抓取与排序
│   ├── L2_creative/      创意：脚本生成、Prompt 编排、锚定
│   ├── L3_visual/        视觉：文生图、图生视频、CLIP 一致性
│   ├── L4_audio/         音频：TTS、混音、时长规划
│   ├── L5_postprod/      后期：音画同步、救援、字幕、封面
│   ├── L6_distribution/  分发：成片回传
│   └── L7_optimization/  优化：质量评分、数据回流
├── scripts/              运维脚本、回填脚本
├── sl-vue/               Vue 3 前端
├── gateway/              Java Spring Boot 网关（见 gateway/README.md）
├── tests/                pytest 测试
├── docs/                 模块文档
├── docker-compose.yaml   多容器编排
├── Dockerfile            Python 镜像
└── main_v2.py            FastAPI 入口
```

## 文档目录

- [`architecture.md`](architecture.md) —— 总体架构与数据流
- [`docs/操作使用手册.md`](docs/操作使用手册.md) —— 操作使用手册：飞书接入 / 测试 / 日常维护（English: [`docs/operations-manual.md`](docs/operations-manual.md)）
- [`docs/启动-停止脚本.md`](docs/启动-停止脚本.md) —— 跨平台启停 / 日志 / 排障指南
- [`gateway/README.md`](gateway/README.md) —— Java 网关（鉴权 / 路由 / failover / 熔断 / 计量）

## 开发

### 跑测试

```bash
docker compose exec -T orchestrator pytest tests/ -q
```

### 修改 Python 代码后

```bash
docker compose build orchestrator
docker compose up -d --force-recreate --no-build orchestrator worker
```

> 注意：`docker compose restart` **不会**重新构建镜像，源代码改动需走上面两步。

### 数据库迁移

```bash
docker compose exec -T orchestrator alembic upgrade head
# 创建新迁移
docker compose exec -T orchestrator alembic revision -m "your message"
```

## 贡献

欢迎提交 Issue 与 Pull Request。建议先通过 Issue 讨论较大改动的设计方向。

## License

MIT —— 详见 [`LICENSE`](LICENSE)。

## 致谢

本项目集成了多个优秀开源模型与商业 AI 服务，包括但不限于：DeepSeek、智谱 GLM、可灵 AI、火山引擎 TTS、faster-whisper、Langfuse、Resilience4j。
