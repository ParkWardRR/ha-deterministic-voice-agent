# Deterministic HA Voice Agent

<div align="center">

[![Rust](https://img.shields.io/badge/Rust-1.80+-000000?logo=rust&style=for-the-badge)](https://www.rust-lang.org/)
[![NVIDIA CUDA](https://img.shields.io/badge/CUDA-Hardware%20Accel-76B900?logo=nvidia&style=for-the-badge)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_v2.0-CUDA/_TensorRT-005CED?logo=onnx&style=for-the-badge)](https://onnxruntime.ai/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX--512-FF5722?style=for-the-badge)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![ARM NEON](https://img.shields.io/badge/ARM_NEON-Hardware%20Accel-0091BD?logo=arm&style=for-the-badge)](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
[![Tokio](https://img.shields.io/badge/Async-Tokio-DCDCDC?style=for-the-badge)](https://tokio.rs/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW%20ANN-4B5563?style=for-the-badge)](https://github.com/pgvector/pgvector)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791?logo=postgresql&style=for-the-badge)](https://www.postgresql.org/)
[![Axum](https://img.shields.io/badge/HTTP-Axum-EF4444?style=for-the-badge)](https://github.com/tokio-rs/axum)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-Custom%20Agent-18BCF2?logo=homeassistant&style=for-the-badge)](https://www.home-assistant.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&style=for-the-badge)](https://www.python.org/)
[![Linux](https://img.shields.io/badge/Linux-x86__64_/_ARM64-FCC624?logo=linux&style=for-the-badge)](https://www.kernel.org/)
[![License](https://img.shields.io/badge/License-Blue_Oak_1.0.0-blue.svg?style=for-the-badge)](https://blueoakcouncil.org/license/1.0.0)

*A deterministic-first voice control orchestrator for Home Assistant written in highly-optimized Rust.*

</div>

## Table of Contents
- [Overview](#overview)
- [Concept Overview (Simple Explanation)](#concept-overview-simple-explanation)
- [Architecture Review](#architecture-review)
- [Request Flow](#request-flow)
- [AI Entity Resolution Process](#ai-entity-resolution-process)
- [System Safety Gates](#system-safety-gates)
- [Technical Deep Dive](#technical-deep-dive)
- [Hardware Specifications](#hardware-specifications)
- [How to Deploy](#how-to-deploy)
  - [Home Assistant Integration Guide](#step-4-home-assistant-integration-guide)
- [Validation & Performance Matrix](#validation--performance-matrix)
- [Project Roadmap](#project-roadmap)
- [Repository Layout](#repository-layout)

---

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
The backend is written entirely in **Rust** using `tokio` for massive multiplexed asynchronous I/O and `axum` for HTTP routing. It is specifically designed for deterministic entity resolution, avoiding hallucination via heavily constrained boundaries. The entire orchestrator daemon requires less than 50MB of baseline RAM utilization.

### 2. Local AI & ONNX Inference
Intent parsing is completely offline and heavily optimized. We embed **Qwen 2.5 (1.5B)** directly into the orchestrator memory space via `ort` (ONNX Runtime v2.0). Using direct CUDA & TensorRT hooks alongside SIMD CPU fallbacks (AVX-512, fp16 computations), intent inference executes in milliseconds. We deploy custom tokenized sequences decoding greedily directly within the event loop, stripping out typical IPC REST latency layers.

### 3. SIMD-Accelerated Vector Retrieval
Device candidate search pairs standard PostgreSQL caching capabilities with ultra-fast vector distance scoring. In-memory exact embeddings mapped from `pgvector` are ranked using heavily unrolled, AVX2 / AVX-512 explicitly vectorized dot-product arithmetic kernels dynamically dispatched via `multiversion` based on CPU detection at startup. **ARM NEON** extensions are natively supported for optimized inference on Apple Silicon and Raspberry Pi 5 workloads.

### 4. Safety Gates & Verification
Before execution, candidate actions must clear strict safety domains. Critical triggers (e.g., locks, garage doors) force active WebSockets confirmation dialogue blocks requiring an exact boolean override back from the Home Assistant frontend, whereas ambiguous intent `candidates[0].score < 0.70` trigger dynamic clarification workflows back to the user prioritizing physical safety.

### 5. Multi-Room Batch Execution
Commands targeting widespread entity clusters natively scale constraints. Using substring heuristics for collective definitions ("all", "every"), the vector parsing ceiling inflates to 20 devices concurrently dynamically. Asynchronous SQL (`sqlx`) database pool connection threads actively adapt dynamically (configurable via `DB_MIN_CONNS` & `DB_MAX_CONNS`) adjusting under massive voice command array workloads efficiently without freezing the node process.

## Hardware Specifications

### Minimum Requirements (CPU inference)
- **CPU**: x86_64 AVX2-capable processor (Intel Haswell / AMD Zen 1 or newer) or ARM64
- **RAM**: ~2.5 GB Total (50MB for Daemon + ~2GB for Qwen 1.5B ONNX Model caching)
- **Storage**: ~2GB disk space for model files

### Recommended Hardware (CUDA acceleration)
- **CPU**: AVX-512 compatible (Intel Skylake-X / AMD Zen 4+)
- **GPU**: NVIDIA GPU with >= 4GB VRAM (e.g., RTX 3050+)
- **RAM**: 8 GB+
- **Database**: PostgreSQL 17 deployed on NVMe SSD for high IOPS

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

### Step 4: Home Assistant Integration Guide

The system utilizes a custom, lightweight Python bridge to route conversational events from Home Assistant's internal speech-to-text pipeline out to our high-performance Rust orchestrator nodes.

1. **Deploy the Component**: Move the Python engine bridge from this repository located at `homeassistant/custom_components/deterministic_agent` into your Home Assistant installations `config/custom_components/` directory.
2. **Reboot Core**: Restart the Home Assistant Core system securely so the registry can aggressively hash and cache the new integration's `manifest.json`.
3. **Register the Hub**: Navigate to **Settings > Devices & Services > Add Integration**, and search for `Deterministic Voice Agent`. Proceed through the UI configuration flow to map your `LISTEN_ADDR` socket.
4. **Pipeline Assignment**: Finally, navigate to **Settings > Voice Assistants**, create or select an active Voice Pipeline, and nominate the newly added `Deterministic Agent` engine as your designated conversation protocol!

**Advanced Features within HA:**
- *Real-time States:* The proxy natively relays `user_input.conversation_id` tokens securely allowing deterministic contextual memory (e.g., executing "turn off the lights" immediately following "turn on the kitchen").
- *Service Execution:* Permitted actions returned by the Rust daemon invoke local `hass.services.async_call` methods physically bypassing external network clouds entirely.

## Validation & Performance Matrix

Tested end-to-end on a live local area network resolving against an `RTX 3060` intent server over Websockets:
- **Fast-Reject/Clarification Latency**: `~160ms` (Sub-second fallback safety queries before TTS kicks in)
- **Generative Execution Latency (V2)**: `~550ms - 570ms` (Full pipeline with Native Scopes & `N=20` Context Limits: Speech parsing -> ONNX LLM CUDA Typed Validation -> Strict JSON plan synthesis -> HA Service Execution)

> 🔗 **See the full latency progression & historical test logs in our [Performance Benchmarks Handbook](benchmarks.md)**.

### Operational Validations
- **Exact Action**: Deterministically matches specific hardware IDs successfully.
- **Ambiguity**: Actively halts inference execution processes to natively request clarification on conflicts.
- **Security Control**: Isolated commands attempting hazardous target interactions are physically halted and blocked.
- **General Queries**: Arbitrary LLM questioning successfully proxy via backend conversational inference routing gracefully.

## Project Roadmap
The agent's feature-set is actively expanding. Key V2 additions (Strict Area Scopes and Typed Entity Budgets) have fundamentally bypassed general unbounded similarity matches. For details on upcoming native Home Assistant integration extensions, and parallel execution logic, refer to the **[Project Roadmap Timeline](roadmap.md)**.

## Repository Layout
- `orchestrator-rs/`: The Rust backend orchestrator engine processing and validation loop APIs.
- `homeassistant/custom_components/`: Home Assistant custom interface proxy payload relay protocols.
