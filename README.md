# myAiVideos

> An end-to-end automated short-video production pipeline. One command in, a publish-ready video out.

**English** | [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

`myAiVideos` is an open-source engineering project for automated Chinese-language short-video production, covering the full pipeline from **topic discovery → creative generation → visuals → audio → post-production → distribution**. It uses FastAPI as the orchestration core and a Java gateway as the platform governance layer (auth / routing / circuit breaker / metering / observability), coordinating multiple AI models for generation and quality gating.

Suitable content formats:

- Hot-news commentary
- Knowledge explainers
- Emotional storytelling
- Curiosity / strange facts
- Social insight clips

## Key Features

- **Seven-layer pipeline architecture**: Topic (L1) → Creative (L2) → Visual (L3) → Audio (L4) → Post-production (L5) → Distribution (L6) → Optimization (L7). Layers are decoupled and independently replaceable.
- **Multi-model failover + circuit breaker**: Java gateway aggregates DeepSeek / Qwen / GLM with Resilience4j protection; failed providers are automatically rotated.
- **AI consistency governance**
  - Prompt anchoring: maintain visual identity of subjects across video segments
  - CLIP text-image consistency gating: reject off-prompt keyframes early
  - Audio-video sync auto-rescue: 4-tier strategy (audio tempo / video pad / narration rewrite)
- **Full-stack observability**: `trace_id` propagated across Java/Python; Langfuse call-tree visibility.
- **Metering & rate limiting**: AOP aspect captures token / cost without code changes, aggregated by tenant.
- **Production-ready**: SSE progress streaming, single-segment regeneration, rolling log archives.

## Tech Stack

| Layer | Components |
|---|---|
| Orchestration (Python) | FastAPI · asyncio · Pydantic · SQLAlchemy (asyncpg) · Alembic · ARQ |
| Gateway (Java) | Spring Boot 3.5 · WebClient · Resilience4j · Caffeine · Prometheus |
| Frontend | Vue 3 · Vite · Vue Flow · Pinia · Tailwind |
| Data | PostgreSQL · Redis · MinIO |
| Observability | Langfuse · cross-language trace context |
| AI Models | DeepSeek · Zhipu GLM-4V · Kling v2.5 · Volcengine / MiniMax TTS · faster-whisper |
| Containers | Docker Compose |

## Architecture

```
                +----------------+
                |  Client / Bot  |
                +--------+-------+
                         |
                +--------v-------+
                |  Java Gateway  |  JWT / TraceId / Router / Breaker / Meter
                +--------+-------+
                         |
                +--------v-------+
                | FastAPI orch.  |  webhook · session · job orchestration
                +--+---------+---+
                   |         |
        +----------v---+   +-v-----------+
        |  ARQ Worker  |   |  7-layer    |
        +-----+--------+   |  pipeline   |
              |            +-+-----------+
              |              |
   +----------v--+    +------v-----------------+
   | Redis(queue)|    | DeepSeek / Kling / TTS |
   +-------------+    +------------------------+
```

For details, see [`architecture.md`](architecture.md).

## Quick Start

### Prerequisites

- Docker Desktop ≥ 24
- Node.js ≥ 20 (only for frontend development)
- Python ≥ 3.11 (only if running locally without Docker, or running tests on host)

### One-Click Start

```bash
# 1. Clone and prepare env
git clone <your-fork>.git
cd myAiVideos
cp .env.example .env
# Edit .env to fill in at least the required AI provider API keys.

# 2. Start backend (postgres + redis + orchestrator + worker)
docker compose up -d postgres redis orchestrator worker

# 3. Run database migrations
docker compose exec -T orchestrator alembic upgrade head

# 4. (Optional) Start Java gateway
docker compose up -d gateway

# 5. Start frontend
cd sl-vue && npm install && npm run dev
```

Health checks:

- Python API: <http://localhost:8000/health>
- Java Gateway: <http://localhost:8080/actuator/health>
- Frontend: <http://localhost:5173>

For Windows / macOS / Linux specific scripts, scenario presets, and troubleshooting, see [`docs/启动-停止脚本.md`](docs/启动-停止脚本.md).

### Configuration

Core environment variables (full list in `.env.example`):

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | Text LLM |
| `GLM_API_KEY` | Multimodal moderation |
| `KLING_ACCESS_KEY` / `KLING_SECRET_KEY` | Text-to-image / image-to-video |
| `ATLASCLOUD_API_KEY` | Optional Atlas Cloud Media API backend when `VISUAL_IMAGE_PROVIDER` or `VISUAL_VIDEO_PROVIDER` is `atlascloud` |
| `VOLCENGINE_TTS_*` or `MINIMAX_TTS_*` | Speech synthesis |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Tracing (optional) |
| `GATEWAY_AUTH_JWT_SECRET` | JWT signing key for Java gateway (≥ 32 bytes) |

## Project Structure

```
myAiVideos/
├── api/                  FastAPI routes (webhook, storyboard, clip)
├── config/               Settings, Skill templates, characters, styles
├── core/                 trace context, langfuse client, logging, scheduler
├── db/                   SQLAlchemy models, Alembic migrations
├── integrations/         External model client wrappers
├── layers/               Seven-layer pipeline
│   ├── L1_trending/      Topic discovery / ranking
│   ├── L2_creative/      Script generation, prompt orchestration, anchoring
│   ├── L3_visual/        Text-to-image, image-to-video, CLIP consistency
│   ├── L4_audio/         TTS, mixing, duration planning
│   ├── L5_postprod/      AV sync, rescue, captions, cover
│   ├── L6_distribution/  Delivery back to client
│   └── L7_optimization/  Quality scoring, data feedback
├── scripts/              Ops & backfill scripts
├── sl-vue/               Vue 3 frontend
├── gateway/              Java Spring Boot gateway (see gateway/README.md)
├── tests/                pytest test suite
├── docs/                 Module reference docs
├── docker-compose.yaml   Multi-container orchestration
├── Dockerfile            Python image
└── main_v2.py            FastAPI entrypoint
```

## Documentation Index

- [`architecture.md`](architecture.md) — Overall architecture and data flow
- [`docs/operations-manual.md`](docs/operations-manual.md) — Operations manual: Feishu integration, testing, daily maintenance (中文: [`docs/操作使用手册.md`](docs/操作使用手册.md))
- [`docs/启动-停止脚本.md`](docs/启动-停止脚本.md) — Cross-platform start / stop / log / troubleshooting guide
- [`gateway/README.md`](gateway/README.md) — Java gateway (auth, routing, failover, circuit breaker, metering)

## Development

### Run tests

```bash
docker compose exec -T orchestrator pytest tests/ -q
```

### After Python code changes

```bash
docker compose build orchestrator
docker compose up -d --force-recreate --no-build orchestrator worker
```

> Note: `docker compose restart` does **not** rebuild the image. Use the two commands above for source code changes.

### Database migrations

```bash
docker compose exec -T orchestrator alembic upgrade head
# Create a new migration
docker compose exec -T orchestrator alembic revision -m "your message"
```

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an issue first to discuss the design.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

This project integrates several excellent open-source models and commercial AI services, including but not limited to: DeepSeek, Zhipu GLM, Kling AI, Volcengine TTS, faster-whisper, Langfuse, Resilience4j.
