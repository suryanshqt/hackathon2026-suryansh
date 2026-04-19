import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_core.messages import HumanMessage
from app.agent.graph import agent_graph
from app.models.schemas import AgentState, TicketState

MONGO_URL    = os.getenv("MONGO_URL", "mongodb://localhost:27017")
TICKETS_PATH = os.path.join(os.path.dirname(__file__), "../data/tickets.json")

CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_client = AsyncIOMotorClient(
        MONGO_URL,
        maxPoolSize=20,
        minPoolSize=2,
        serverSelectionTimeoutMS=5000,
    )
    app.state.db = mongo_client.shopwave_db
    app.state.logs_collection = app.state.db.audit_logs
    app.state.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    yield
    mongo_client.close()

app = FastAPI(title="Hackathon Ksolves", lifespan=lifespan)


def load_tickets() -> list[dict]:
    with open(TICKETS_PATH, "r") as f:
        return json.load(f)


def serialize_message(m) -> dict:
    """
    Safely serialize a message — handles both LangChain message objects
    and raw dicts, preventing 'dict has no attribute content' crashes.
    """
    if isinstance(m, dict):
        return {
            "role": m.get("type", m.get("role", "unknown")),
            "content": m.get("content", ""),
            "tool_calls": m.get("tool_calls", None),
        }
    return {
        "role": getattr(m, "type", "unknown"),
        "content": getattr(m, "content", ""),
        "tool_calls": getattr(m, "tool_calls", None),
    }


def _get_final_status(messages: list) -> str:
    from langchain_core.messages import ToolMessage
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "") or ""
            if name == "send_reply":
                return "resolved"
            if name == "escalate":
                return "escalated"
    return "unknown"


async def process_single_ticket(ticket: dict, logs_collection, semaphore: asyncio.Semaphore) -> dict:
    ticket_id = ticket.get("ticket_id")

    # Force print to bypass Docker buffer
    print(f"Queued {ticket_id}...", flush=True)

    async with semaphore:
        print(f"Processing {ticket_id}...", flush=True)
        
        initial_state = AgentState(
            messages=[HumanMessage(content=f"Subject: {ticket['subject']}\n\n{ticket['body']}")],
            ticket_state=TicketState(ticket_id=ticket_id)
        )
        try:
            final_state = None
            for attempt in range(5):
                try:
                    final_state = await agent_graph.ainvoke(initial_state)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        wait = 60 * (attempt + 1)
                        # Ensure rate limit logs print immediately
                        print(f"[{ticket_id}] Rate limited, waiting {wait}s (attempt {attempt+1}/5)", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    raise

            if final_state is None:
                raise Exception("Max retries exceeded")

            audit_entry = {
                "ticket_id": ticket_id,
                "classification": final_state["ticket_state"].classification.model_dump(),
                # Use serialize_message to handle both dict and LangChain objects
                "conversation_history": [
                    serialize_message(m) for m in final_state["messages"]
                ],
                "final_status": _get_final_status(final_state["messages"]),
            }

            await logs_collection.update_one(
                {"ticket_id": ticket_id},
                {"$set": audit_entry},
                upsert=True,
            )
            return {"ticket_id": ticket_id, "status": "success"}

        except Exception as e:
            await logs_collection.update_one(
                {"ticket_id": ticket_id},
                {"$set": {"ticket_id": ticket_id, "final_status": "error", "error": str(e)}},
                upsert=True,
            )
            return {"ticket_id": ticket_id, "status": "failed", "error": str(e)}


@app.post("/process-all")
async def process_all_tickets():
    """
    Processes all tickets concurrently, bounded by CONCURRENCY_LIMIT.
    Uses return_exceptions=True so one failing ticket never kills the batch.
    """
    tickets = load_tickets()
    semaphore       = app.state.semaphore
    logs_collection = app.state.logs_collection

    tasks = [process_single_ticket(t, logs_collection, semaphore) for t in tickets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    normalized = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            normalized.append({"ticket_id": tickets[i].get("ticket_id"), "status": "failed", "error": str(r)})
        else:
            normalized.append(r)

    success_count = sum(1 for r in normalized if r.get("status") == "success")
    return {
        "processed": len(normalized),
        "succeeded": success_count,
        "failed": len(normalized) - success_count,
        "results": normalized,
    }


@app.get("/audit-log")
async def get_audit_log():
    """Returns the full audit trail from MongoDB."""
    cursor = app.state.logs_collection.find({}, {"_id": 0})
    logs = await cursor.to_list(length=100)
    return logs