# Failure Mode Analysis & System Guardrails
**ShopWave AI Support Agent — Production Resilience Documentation**

---

## Overview

This document details **4 critical failure modes** identified and handled in the ShopWave agentic support system, built on LangGraph + LangChain + OpenAI. Each failure mode is grounded in the actual production code and describes the exact mechanism by which the system detects, contains, and recovers from the failure without human intervention or data loss.

---

## Failure Mode 1: Concurrency-Induced API Rate Limiting (HTTP 429)

### Scenario
When processing a batch of 20 support tickets concurrently, the OpenAI API rejects requests with a `429 Too Many Requests` error because token-per-minute (TPM) quotas are exhausted. Standard API tiers typically cap at ~30,000 TPM, which is overwhelmed when many tickets are processed in parallel — each requiring multiple LLM calls for classification, reasoning, tool invocation, and reply generation.

### How It Happens
Each ticket spins up its own LangGraph agent instance. Without rate control, all 20 agents attempt to call the LLM simultaneously, flooding the endpoint.

### System Response

**1. Semaphore-Based Traffic Control (`main.py`)**  
An `asyncio.Semaphore` limits the number of in-flight LangGraph agent instances at any point in time, ensuring the burst of concurrent API calls stays within quota boundaries.

**2. Client-Level Exponential Backoff (`main.py`)**  
The `ChatOpenAI` client is configured with `max_retries=7`. An outer `try/except` loop detects `429` responses and applies an escalating sleep: `60 * attempt` seconds, giving the provider time to reset the token window before retrying.

**3. Simulated Latency in Tools (`kb_tool.py`, `read_tools.py`, `write_tools.py`)**  
All tools include `await asyncio.sleep(random.uniform(...))` calls. This isn't merely decorative — it ensures async tasks yield control back to the event loop, allowing other coroutines to make progress and preventing a single ticket from monopolizing the thread during I/O waits.

```python
# kb_tool.py — forces event loop yield during KB lookup
await asyncio.sleep(random.uniform(0.1, 0.2))

# Async OpenAI client prevents blocking during embedding generation
client = AsyncOpenAI()
response = await client.embeddings.create(input=query, model="text-embedding-3-small")

# Synchronous Qdrant search offloaded to thread pool
results = await asyncio.to_thread(search, query_vector=query_vector, top_k=3)
```

### Outcome
The batch processes all 20 tickets without crashing, data loss, or permanent failure. Tickets that hit rate limits are paused and retried; the semaphore ensures the system self-regulates without requiring operator intervention.

---

## Failure Mode 2: Unreliable Upstream Tool Responses (Timeouts & Malformed Data)

### Scenario
Simulated internal microservices — `get_order`, `get_customer` — intermittently return a `TimeoutError` or `MalformedResponse`. The payment processor (`issue_refund`) can return a `PaymentGatewayError` (504 Gateway Timeout). These simulate real-world database instability and third-party processor downtime.

### How It Happens
Probability-weighted failure injection is baked directly into the tool implementations:

```python
# read_tools.py — 10% malformed response on get_customer
if random.random() < 0.10:
    return {
        "success": False,
        "error": "MalformedResponse: Data corrupted during transit. Please retry.",
    }

# read_tools.py — 15% timeout on get_order
if random.random() < 0.15:
    return {
        "success": False,
        "error": "TimeoutError: Database failed to respond. You must retry this tool call.",
    }

# write_tools.py — 10% payment gateway failure on issue_refund
if random.random() < 0.10:
    return {
        "success": False,
        "retryable": True,
        "error": "PaymentGatewayError: 504 Gateway Timeout. Retry up to 3 times...",
    }
```

### System Response

**1. Error-as-Observation Pattern**  
Tools never raise Python exceptions to the agent. Instead, all failure cases return a structured dict with `"success": False` and a descriptive `"error"` string. The LangGraph agent receives this as an observation in its ReAct loop and can reason about it explicitly.

**2. Agent-Level Retry with `retryable` Flag**  
The `AGENT_SYSTEM_PROMPT` instructs the LLM to inspect the `retryable` field. For `PaymentGatewayError` where `retryable=True`, the agent autonomously retries `issue_refund` up to **3 times** before abandoning the path.

**3. Safe Escalation After Retry Exhaustion**  
If all 3 retries fail, the agent calls `escalate()` with `priority='high'` rather than leaving the ticket unresolved. The `recommended_path` field in the escalation payload communicates the failure context to the human specialist.

```python
# write_tools.py — escalate validates all required fields before routing
if priority not in VALID_PRIORITIES:
    return {"success": False, "error": f"Invalid priority '{priority}'. Must be one of: ..."}
```

**4. Knowledge Base Fault Isolation (`kb_tool.py`)**  
The entire embedding + vector search flow is wrapped in a `try/except`. Any failure (OpenAI API down, Qdrant unreachable, malformed vector) returns a graceful error string rather than crashing the LangGraph node:

```python
except Exception as e:
    return f"KnowledgeBaseError: Failed to retrieve policies due to {str(e)}. Please try a different query or escalate."
```

### Outcome
No tool failure propagates as an unhandled exception. The agent retries recoverable errors, escalates unrecoverable ones, and always terminates the ticket with either a `send_reply` or `escalate` call.

---

## Failure Mode 3: Context-Poor Customer Inquiries (Zero Identifying Information)

### Scenario
A customer submits a vague ticket with no actionable identifiers — no email, no order ID, no product name. Examples: *"I want a refund"*, *"my thing is broken"*, *"you guys messed up my order"*.

### How It Happens
Common human behaviour in support channels. Customers assume the company already knows who they are.

### System Response

**1. Zero-Hallucination Policy (System Prompt)**  
The `AGENT_SYSTEM_PROMPT` explicitly prohibits the agent from:
- Assuming an email address
- Inventing or guessing an Order ID
- Using placeholder values like `ORDER_ID_HERE`

**2. Tool-Level Guard: No Lookup Without Identifier**  
Attempting to call `get_order` or `get_customer` without real identifiers would immediately return a `"No customer found"` error. The agent is trained to recognise when it lacks the inputs required for any lookup tool and to skip them entirely.

**3. Direct Escalation with Structured Summary**  
Rather than looping through failing tool calls, the agent immediately routes to `escalate()`. The `issue_summary` field is populated with exactly what the customer said and explicitly notes which identifying information is absent:

```python
# write_tools.py — escalate enforces non-empty summary
if not issue_summary:
    return {"success": False, "error": "issue_summary cannot be empty — describe what the issue is."}
if not recommended_path:
    return {"success": False, "error": "recommended_path cannot be empty — state what the human agent should do."}
```

**4. Human-Readable Handoff**  
The escalation payload includes `handoff_summary` and `recommended_path`, saving the human agent from re-reading the raw ticket. They receive a pre-triaged summary explaining what is missing and what to do next.

### Outcome
Vague tickets never loop indefinitely or result in a hallucinated resolution. They are immediately escalated with enough context for a human to resolve them on first contact.

---

## Failure Mode 4: Duplicate Refund / State Mutation Visibility

### Scenario
After a refund is successfully issued, a second agent instance (or a retry loop) attempts to call `issue_refund` again for the same order — either due to a transient network error masking the success response, or due to stale in-memory state where `read_tools.py` doesn't see the mutation made by `write_tools.py`.

### How It Happens
In earlier versions of the code, each file loaded its own copy of the JSON data from disk. A mutation in `write_tools.py` (setting `refund_status = "refunded"`) was invisible to a subsequent `get_order` call in `read_tools.py`, since it was reading from its own independent copy.

### System Response

**1. Single Shared In-Memory State (`shared_data.py`)**  
Both `read_tools.py` and `write_tools.py` import their data from the same `shared_data.py` module. Python's module system guarantees this is a single object in memory — mutations in one file are instantly visible in the other:

```python
# shared_data.py — single source of truth loaded once at startup
CUSTOMERS_DB: list[dict] = _load("customers.json")
ORDERS_DB:    list[dict] = _load("orders.json")
PRODUCTS_DB:  list[dict] = _load("products.json")

# write_tools.py — mutates the shared list in-place
order["refund_status"] = "refunded"
order["status"]        = "refunded"
```

**2. Multi-Layer Duplicate Refund Blocking**  
The guard appears at **three independent levels**, each of which independently blocks a duplicate:

- **`check_refund_eligibility` (`read_tools.py`)** — returns `eligible=False` with reason `"Order has already been refunded"` before the agent even considers calling `issue_refund`.
- **`issue_refund` (`write_tools.py`)** — checks `refund_status == "refunded"` and returns `retryable=False` error, preventing the payment gateway call.
- **`check_return_window` (`read_tools.py`)** — independently surfaces `within_window=False` with `refund_status: "refunded"` as the reason.

```python
# write_tools.py — hard block on duplicate refund
if order.get("refund_status") == "refunded":
    return {
        "success": False,
        "retryable": False,
        "error": f"Order {order_id} has already been refunded. Duplicate refund blocked.",
    }
```

**3. MongoDB Upsert Idempotency (`mongo.py`)**  
Audit log and resolution persistence use `update_one` with `upsert=True` rather than `insert_one`. This means a retry of the same ticket write is idempotent — it updates the existing record rather than creating a duplicate audit entry:

```python
# mongo.py — idempotent write prevents duplicate audit logs on retry
await audit_collection.update_one(
    {"ticket_id": audit.get("ticket_id")},
    {"$set": audit},
    upsert=True
)
```

### Outcome
Duplicate refunds are structurally impossible within a single process run. The three-layer guard means even if the agent retries due to a gateway error masking a successful response, the second attempt is silently blocked with a clear, non-retryable error, and the ticket is escalated rather than double-charged.

---

## Summary Table

| # | Failure Mode | Detection Point | Recovery Mechanism | Data Safety |
|---|---|---|---|---|
| 1 | API Rate Limiting (429) | `main.py` outer loop | Semaphore + exponential backoff | No ticket dropped |
| 2 | Tool Timeouts & Gateway Errors | Tool return value `success=False` | Agent retry (3x max) then escalate | No silent failure |
| 3 | Vague / Context-Poor Tickets | Agent reasoning + tool guard | Immediate escalation with summary | No hallucinated resolution |
| 4 | Duplicate Refund Attempts | `shared_data.py` + 3-layer guard | Non-retryable block + escalation | Payment idempotency enforced |

---

*Generated for the ShopWave AI Support Agent — Ksolves En(AI)bling Hackathon 2026 submission.*
