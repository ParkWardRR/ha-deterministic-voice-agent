# Deterministic HA Voice Agent

<div align="center">

[![Rust](https://img.shields.io/badge/Rust-1.80+-000000?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)
[![Home%20Assistant](https://img.shields.io/badge/Home%20Assistant-Custom%20Conversation%20Agent-18BCF2?logo=homeassistant)](https://www.home-assistant.io/)
[![Postgres](https://img.shields.io/badge/PostgreSQL-17-336791?logo=postgresql)](https://www.postgresql.org/)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-v2.0-005CED?logo=onnx)](https://onnxruntime.ai/)
[![License: Blue Oak 1.0.0](https://img.shields.io/badge/License-Blue_Oak_1.0.0-blue.svg)](https://blueoakcouncil.org/license/1.0.0)

*A deterministic-first voice control orchestrator for Home Assistant written in highly-optimized Rust.*

</div>

## Overview

Unlike standard smart home voice assistants that rely entirely on large, unpredictable generative models, this system provides a **deterministic-first** approach. 

1. **Voice Input**: You give a command (e.g., "turn on the basement stairs light").
2. **Deterministic Resolution**: The system queries a local PostgreSQL/pgvector database using fast lexical and vector similarity searches to identify the exact target devices.
3. **Intent Parsing**: A highly-optimized local AI model translates the command into a strict JSON execution plan.
4. **Safety Verification**: Security rules are applied. Dangerous commands are blocked or prompt for confirmation.
5. **Execution**: The sanitized plan is safely executed via Home Assistant service calls. Questions unrelated to home automation automatically fall back to a general conversational LLM.

The result is extremely high accuracy and low latency without randomly guessing unintended actions.

## Concept Overview (Simple Explanation)

Think of this like a smart home referee:
1. You say: "turn on basement lights".
2. The referee checks a local list of your devices (fast, deterministic).
3. A small model picks the exact action from those candidates only.
4. Safety rules block dangerous stuff or ask follow-up questions.
5. If your question is not about devices ("what's the weather?"), it sends it to a chat model.

Result: fewer wrong-device actions than "just ask one giant LLM to guess everything."

## Architecture Review

![Cloud Architecture](assets/architecture.png)

## Request Flow

![Sequence Flow](assets/request_flow.png)

## AI Entity Resolution Process

![Entity Resolution](assets/entity_resolution.png)

## System Safety Gates

![Safety Flow](assets/safety_gate.png)

## Technical Deep Dive

This orchestrator is engineered for production-grade scale, speed, and safety.

### 1. High-Performance Rust Core
The backend is written entirely in **Rust** using `tokio` for massive asynchronous I/O and `axum` for HTTP routing. It is specifically designed for deterministic entity resolution, avoiding hallucination via heavily constrained constraints.

### 2. Local AI & ONNX Inference
Intent parsing is completely offline and heavily optimized. We embed **Qwen 2.5 (1.5B)** directly into the orchestrator memory space via `ort` (ONNX Runtime v2.0). Using direct CUDA & TensorRT hooks alongside SIMD CPU fallbacks (AVX-512, f16 operations), intent inference executes in milliseconds without HTTP latency overhead.

### 3. SIMD-Accelerated Vector Retrieval
Device candidate search pairs standard PostgreSQL caching capabilities with ultra-fast vector distance scoring. In-memory embeddings mapped from `pgvector` are ranked using heavily unrolled, AVX2 / AVX-512 explicitly vectorized dot-product arithmetic kernels dynamically dispatched via `multiversion`. 

### 4. Safety Gates & Verification
Before execution, candidate actions must clear strict safety domains. Critical triggers (e.g., locks, garage doors) will force explicit confirmation dialogue blocks, and unclear commands with `score < 0.70` trigger dynamic clarification intents back to the user to prevent erroneous state mutations.

### 5. Multi-Room Batch Execution
Commands targeting widespread entity clusters natively expand constraints (e.g. up to 20 devices parsed concurrently) when explicit phrases ("all", "every") are detected. Asynchronous SQL database pool connections actively tunable up and down directly under massive Voice Command payloads.

## How to Deploy

### Prerequisites
- A modern Linux server (x86_64 or ARM64)
- PostgreSQL 17 with the `pgvector` extension
- Home Assistant core instance

### Step 1: Install the Orchestrator Daemon
1. Clone the repository and build the server binary natively:
```bash
git clone https://github.com/ParkWardRR/ha-deterministic-voice-agent
cd ha-deterministic-voice-agent/orchestrator-rs
cargo build --release
```
2. Move the built executable (`target/release/orchestrator`) into your `/opt/` or global bin directory.

### Step 2: Config Environment Setup
Define your explicit integration mapping via `.env` parameter configurations to customize DB size limits:
```ini
LISTEN_ADDR=0.0.0.0:5000
PG_DSN=postgres://agent:password@localhost:5432/agent
HA_URL=http://homeassistant.local:8123
HA_TOKEN=your_long_lived_ha_access_token
MODEL_DIR=/path/to/onnx/weights
DB_MIN_CONNS=2
DB_MAX_CONNS=20
```

### Step 3: Run the Service
Ensure the daemon remains running smoothly via `systemd` process managers:
```bash
sudo cp systemd/zagato-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zagato-agent.service
```

### Step 4: Home Assistant Integration
1. Move the Python engine bridge from `homeassistant/custom_components/deterministic_agent` into your Home Assistant installations `config/custom_components/` directory.
2. Restart Home Assistant Core securely to apply.
3. Navigate to **Settings > Voice Assistants**, designate the new Deterministic Agent engine, and allocate it as your core Pipeline backend conversation protocol!

## Validation Matrix

- **Exact Action**: Deterministically matches specific hardware IDs successfully.
- **Ambiguity**: Actively halts inference execution processes to natively request clarification on conflicts.
- **Security Control**: Isolated commands attempting hazardous target interactions are physically halted and blocked.
- **General Queries**: Arbitrary LLM questioning successfully proxy via backend conversational inference routing gracefully.

## Repository Layout
- `orchestrator-rs/`: The Rust backend orchestrator engine processing and validation loop APIs.
- `homeassistant/custom_components/`: Home Assistant custom interface proxy payload relay protocols.
