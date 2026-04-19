# ShopWave AI Support Agent

**Ksolves En(AI)bling Hackathon 2026**

An autonomous customer support agent built on LangGraph and LangChain that processes e-commerce support tickets end-to-end — classifying intent, looking up order and customer data, applying refund and return policies, and issuing resolutions without human intervention.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Agent](#running-the-agent)
- [Architecture Summary](#architecture-summary)
- [Tools Reference](#tools-reference)
- [Environment Variables](#environment-variables)
- [Submission Checklist](#submission-checklist)

---

## Overview

The ShopWave AI Support Agent autonomously handles 20 mock customer support tickets covering:

- Refund requests (within and outside return window)
- Order cancellations
- Warranty claims
- General policy and FAQ inquiries
- Context-poor or vague customer messages

The agent uses a ReAct reasoning loop within a LangGraph state machine. It looks up customer and order data, checks policy eligibility via a Qdrant-backed knowledge base, executes or denies actions, and always terminates with either a `send_reply` or `escalate` tool call. No ticket is left unresolved.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph + LangChain |
| LLM | OpenAI GPT (ChatOpenAI) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant |
| Database | MongoDB (via Motor async client) |
| Async runtime | Python asyncio |
| Data validation | Pydantic v2 |

---

## Project Structure

```
shopwave-agent/
├── app/
│   ├── agent/
│   │   ├── graph.py              # LangGraph state machine and node definitions
│   │   ├── nodes.py              # Classifier, resolver, and terminal node logic
│   │   └── prompts.py            # AGENT_SYSTEM_PROMPT and classification prompts
│   ├── tools/
│   │   ├── read_tools.py         # get_customer, get_order, check_refund_eligibility, check_warranty
│   │   ├── write_tools.py        # issue_refund, cancel_order, escalate, send_reply
│   │   └── kb_tool.py            # search_knowledge_base (Qdrant RAG)
│   └── db/
│       ├── shared_data.py        # Single in-memory copy of customers, orders, products
│       ├── mongo.py              # Async MongoDB client for audit logs and resolutions
│       └── qdrant_client.py      # Qdrant collection management and vector search
├── data/
│   ├── knowledge-base.md         # ShopWave policy and FAQ source document
│   ├── customers.json            # Mock customer profiles
│   ├── orders.json               # Mock order records
│   ├── products.json             # Mock product catalog
│   └── tickets.json              # 20 mock support tickets
├── scripts/
│   └── ingest.py                 # One-time KB ingestion script (chunk, embed, upsert)
├── main.py                       # Entry point — concurrent batch processing of all tickets
├── docker-compose.yml            # Qdrant + MongoDB services
├── requirements.txt
├── .env.example
├── architecture.pdf              # Agent loop and tool design diagram
├── failure_modes.md              # Documented failure scenarios and system responses
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An OpenAI API key

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/hackathon2026-<your-name>.git
cd hackathon2026-<your-name>
```

### Step 2 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values (see [Environment Variables](#environment-variables) below).

### Step 3 — Start infrastructure services

```bash
docker compose up -d
```

This starts Qdrant (port 6333) and MongoDB (port 27017).

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Ingest the knowledge base

This step chunks `data/knowledge-base.md`, embeds each chunk using `text-embedding-3-small`, and upserts the vectors into Qdrant. Run it once before the first agent run.

```bash
python -m scripts.ingest
```

Expected output:
```
── ShopWave KB Ingestion ──
Creating Qdrant collection if needed...
Chunking .../knowledge-base.md...
  → N chunks created
Embedding N chunks...
Upserting into Qdrant...
Done! N vectors stored in Qdrant.
```

---

## Running the Agent

### Process all 20 tickets

```bash
python main.py
```

The agent processes all tickets concurrently (bounded by an `asyncio.Semaphore` to respect API rate limits). Results are written to MongoDB and an audit log is generated.

### Output files

| File | Description |
|---|---|
| `audit_log.json` | Tool calls, reasoning traces, and decisions for all 20 tickets |
| MongoDB `tickets` collection | Full resolution records per ticket |
| MongoDB `audit_logs` collection | Append-only audit trail |

To export the audit log to a local file after the run:

```bash
python -c "
import asyncio, json
from app.db.mongo import get_all_audit_logs
logs = asyncio.run(get_all_audit_logs())
with open('audit_log.json', 'w') as f:
    json.dump(logs, f, indent=2)
print(f'Exported {len(logs)} audit log entries.')
"
```

---

## Architecture Summary

The agent follows a **ReAct (Reason + Act)** loop inside a LangGraph state machine.

```
Ticket Input
     |
     v
[Classifier Node]  — Categorises ticket: refund / cancel / warranty / info / exchange
     |
     v
[Resolver Node]    — ReAct loop: reasons, calls tools, reads observations, repeats
     |             — Max iterations enforced to prevent infinite loops
     v
[Terminal Guard]   — Intercepts any execution reaching END without a terminal tool call
     |             — Forces send_reply or escalate using last reasoning trace
     v
Resolution (send_reply) or Escalation (escalate)
     |
     v
[Audit Node]       — Persists full tool call trace and resolution to MongoDB
```

**State object (`AgentState`)** carries the full message history and a `TicketResolution` Pydantic model through every node, ensuring type safety and clean serialisation to MongoDB.

**Shared in-memory data** (`shared_data.py`) ensures that write mutations (e.g. `issue_refund` marking an order as `refunded`) are immediately visible to subsequent read tool calls within the same process, preventing stale-state reads.

---

## Tools Reference

### Read Tools (`read_tools.py`)

| Tool | Purpose |
|---|---|
| `get_customer` | Look up customer profile and tier by email |
| `get_order` | Look up order by order ID |
| `get_product` | Look up product details by product ID |
| `get_orders_by_customer_email` | Find all orders for a customer when no order ID is provided |
| `check_return_window` | Check if an order is within its return deadline |
| `check_refund_eligibility` | Full eligibility check before any refund action |
| `check_warranty` | Check if the product is within its warranty period |

### Write Tools (`write_tools.py`)

| Tool | Purpose |
|---|---|
| `issue_refund` | Issues a refund to the original payment method (irreversible) |
| `cancel_order` | Cancels a processing-status order |
| `escalate` | Routes ticket to human specialist with structured handoff summary |
| `send_reply` | Sends the final customer-facing resolution email |

### Knowledge Base Tool (`kb_tool.py`)

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Semantic search over ShopWave policy and FAQ documents via Qdrant |

---

## Environment Variables

Copy `.env.example` to `.env` and populate the following:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB=shopwave
```

No API keys are hardcoded anywhere in the repository. All secrets are loaded via `python-dotenv` at runtime.

---

## Submission Checklist

| Deliverable | File | Status |
|---|---|---|
| Working agent with entry point | `main.py` | Included |
| README with setup instructions | `README.md` | This file |
| Architecture diagram | `architecture.pdf` | Included |
| Failure mode analysis | `failure_modes.md` | Included |
| Audit log (all 20 tickets) | `audit_log.json` | Generated on run |
| Recorded demo | `demo.mp4` | Included |

---

*ShopWave AI Support Agent — Ksolves En(AI)bling Hackathon 2026*
