<div align="center">

# ⚡ Kyo

**Ultra-fast 140M System-One Decision Engine for AI Agents & Guardrails**

[![PyPI version](https://img.shields.io/pypi/v/kyo.svg?color=blue)](https://pypi.org/project/kyo/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-open--zzrl%2Fkyo-yellow)](https://huggingface.co/open-zzrl/kyo)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

*Sub-10ms deterministic routing, policy enforcement, and security gating for agentic workflows.*

</div>

---

## What is Kyo?

**Kyo** is an ultra-compact (140M parameter) **System-One** reasoning engine engineered to sit in front of autonomous LLM pipelines. Instead of wasting expensive frontier tokens (Claude 3.5 Sonnet, GPT-4o) on boolean policy checks, risk evaluations, and deterministic tool routing, Kyo evaluates raw state dumps, telemetry logs, and JSON payloads in **sub-10ms** with **100% order-invariance**.

* **Backbone:** Bidirectional Transformer (`mmBERT-small`, 140M parameters, hidden size 384).
* **Architecture:** Pairwise Cross-Encoder with **Dual Pooling** (`[CLS]` + Masked Mean $\to$ 768d).
* **100% Order-Invariance:** Candidate options are evaluated in isolated pairwise cross-attention passes, completely eliminating letter/position biases.
* **Calibrated Confidence:** Built-in thresholding automatically flags uncertain decisions for escalation to frontier models.

---

## Honest Benchmark Results

Evaluated on the independent test splits of standard decision benchmarks:

### 1. Agent Policies & Guardrails (`LocalLLaMA/typed-decisions`, 2,000 hold-out scenarios)

| Metric / Benchmark | Kyo (140M) | Laya (421M) | TypeSafe Jev |
| :--- | :--- | :--- | :--- |
| Typed-decisions (Overall) | 72.90% | **76.60%** | 72.70% |
| ↳ noul (Boolean Gating) | 84.00% | **85.70%** | 77.50% |
| ↳ choice (Action / Tool Routing) | 70.00% | **73.30%** | 72.00% |
| ↳ score (Risk Tiers 0-3)) | 66.75% | **72.30%** | 69.60% |
| Order-Invariance | **100.00%** | 27.05% | 13.0% |

### 2. Multi-Column Tabular Arithmetic (`avbiswas/bev-decision`, 1,760 hold-out samples)

| Benchmark | Kyo (140M) | Laya Base (149M) | Jev-0.5B (490M) |
|---|:---:|:---:|:---:|
| **bev-decision (holdout)** | **57.67%** | 71.80% | 75.40% |

> **Scope Note:** Kyo is specialized for **Agent Policy Enforcement, Security Guardrails, and Action Selection**. For heavy multi-column relational table arithmetic requiring large numerical working memory (such as `bev-decision`), larger models (Laya-Large / Qwen / Frontier LLMs) are recommended.

---

## Architecture: Dual-Pooling Cross-Encoder

Standard bi-encoders discard token interactions, while causal autoregressive decoders suffer from positional bias (preferring option A over B). Kyo uses **Pairwise Cross-Attention**:


```
[Context + Instruction] ──┐
├──> [mmBERT-small (12 layers)] ──> [CLS (384)] ────────┐
[Candidate Option (k)] ───┘                                   [Mean Pool (384)] ──┴─> LayerNorm (768) ──> Classifier ──> Score_k
```

1. **Independent Evaluation:** Each `(Context, Option_k)` pair is evaluated without token contamination from other candidate options.
2. **Dual Representation:** Concatenates global semantic representations (`[CLS]`) with token-level sequence representations (`Masked Mean Pooling`), expanding feature capacity to 768 dimensions without adding backbone parameters.

---

## Installation

```bash
pip install kyo
```

Or install from source in editable mode:

```bash
git clone [https://github.com/open-zzrl/kyo.git](https://github.com/open-zzrl/kyo.git)
cd kyo
pip install -e .
```

---

## Quickstart

```python
import torch
from kyo import Kyo

# 1. Load the engine directly from Hugging Face Hub
engine = Kyo.from_pretrained("open-zzrl/kyo", confidence_threshold=0.85)

telemetry = {
    "agent_id": "payment-worker-4",
    "endpoint": "/api/v1/transfer/batch",
    "requested_amount_usd": 145000,
    "spending_limit_usd": 50000,
    "user_approval_present": False,
    "geo_anomaly": True
}

instruction = "Verify transaction safety against corporate treasury compliance rules."

options = {
    "Allow": "Transaction within limits and conforms to standard policy.",
    "Require2FA": "Minor anomaly; hold for secondary automated factor.",
    "BlockAndEscalate": "Hard limit breach or suspicious telemetry; freeze and alert human."
}

# 2. Warm-up (initializes CUDA context and JIT-compiles SDPA kernels)
_ = engine.decide(context=telemetry, instruction=instruction, options=options)

# 3. Benchmark run (Hot inference)
result = engine.decide(
    context=telemetry,
    instruction=instruction,
    options=options
)

print("-" * 55)
print(f"Device             : {next(engine.model.parameters()).device}")
print(f"Decision           : {result.decision}")
print(f"Confidence         : {result.confidence * 100:.2f}%")
print(f"HOT Latency        : {result.latency_ms:.2f} ms")
print(f"Requires Fallback  : {result.fallback_to_llm}")
print("Probability Scores :")
for opt, score in result.scores.items():
    print(f"  - {opt:<18}: {score * 100:>6.2f}%")
print("-" * 55)
```

### Accessing Scored Distributions

```python
print(f"Chosen Action : {result.decision}")
print(f"Confidence    : {result.confidence * 100:.2f}%")
print(f"Latency       : {result.latency_ms:.2f} ms")
print("Option Distribution:")
for key, prob in result.scores.items():
    print(f"  - {key}: {prob * 100:.1f}%")
```

---

## Developer Guide

### Smoke Test

Run the verification test against local checkpoints or remote weights:

```bash
python test_inference.py
```

### Uploading Checkpoints to Hugging Face Hub

```bash
huggingface-cli login
python push_to_hub.py
```

### Packaging & PyPI Release

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## License

This project is licensed under the [Apache-2.0 License](https://www.google.com/search?q=LICENSE).
