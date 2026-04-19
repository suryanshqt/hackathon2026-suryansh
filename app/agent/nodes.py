# app/agent/nodes.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from app.models.schemas import AgentState, TicketClassification
from app.tools.read_tools import (
    get_customer,
    get_order,
    get_product,
    get_orders_by_customer_email,
    check_return_window,
    check_refund_eligibility,   # required by hackathon spec — explicit tool
    check_warranty,
)
from app.tools.write_tools import issue_refund, cancel_order, escalate, send_reply
from app.tools.kb_tool import search_knowledge_base
from app.agent.prompts import CLASSIFICATION_PROMPT, AGENT_SYSTEM_PROMPT

# ─── Tool Registry ────────────────────────────────────────────────────────────
# Order matters for the LLM's tool selection — least destructive first.
# Read tools come before write tools so the agent naturally looks before it acts.

ALL_TOOLS = [
    # ── Read / lookup ──────────────────────────────────────────────────────
    get_customer,               # customer profile, tier, notes
    get_order,                  # order status, delivery date, amount
    get_product,                # product category, warranty, return window
    get_orders_by_customer_email,  # find orders when no order ID given
    check_return_window,        # is order within return window?
    check_refund_eligibility,   # REQUIRED by hackathon spec — eligibility + reason
    check_warranty,             # is product under warranty?
    # ── Knowledge base ─────────────────────────────────────────────────────
    search_knowledge_base,      # semantic search over policy & FAQ docs
    # ── Write / act (irreversible — ordered least to most destructive) ─────
    cancel_order,               # cancel a processing order
    issue_refund,               # IRREVERSIBLE — refund to payment method
    escalate,                   # hand off to human with structured summary
    send_reply,                 # send final email to customer
]

# ─── LLM Setup ────────────────────────────────────────────────────────────────

# max_retries=7 gracefully handles OpenAI 30k TPM rate limits under concurrency.
# LangChain will automatically back off and retry instead of crashing the ticket.
_base_llm  = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=7)
triage_llm = _base_llm.with_structured_output(TicketClassification)
agent_llm  = _base_llm.bind_tools(ALL_TOOLS)

# ─── Tool Node ────────────────────────────────────────────────────────────────

tool_node = ToolNode(ALL_TOOLS)

# ─── Triage Node ─────────────────────────────────────────────────────────────

async def triage_ticket(state: AgentState) -> dict:
    """
    Classifies the incoming ticket into a structured TicketClassification.

    Uses with_structured_output so LangChain handles Pydantic parsing and
    validation automatically. The result is stored in ticket_state.classification
    and injected into the agent system prompt on every loop iteration.

    Output fields:
      - category: refund | return | cancel | warranty | exchange | info
      - urgency:  low | medium | high | urgent
      - resolvable: bool
      - confidence: 0.0–1.0
      - reasoning: str
    """
    ticket_content = state["messages"][0].content

    classification: TicketClassification = await triage_llm.ainvoke([
        SystemMessage(content=CLASSIFICATION_PROMPT),
        # Explicitly using HumanMessage prevents strict-validation crashes
        # that can occur if the content is passed as a raw string
        HumanMessage(content=ticket_content),
    ])

    state["ticket_state"].classification = classification
    return {"ticket_state": state["ticket_state"]}


# ─── Agent Node ───────────────────────────────────────────────────────────────

async def resolve_ticket(state: AgentState) -> dict:
    """
    Core ReAct reasoning loop.

    Called on every iteration of the agent loop — after triage and after each
    tool result. Injects the triage classification (category, urgency, confidence)
    into the system prompt so context is never lost across loop iterations.

    The agent reasons step-by-step, selects which tool to call next, and
    continues until it calls send_reply or escalate (terminal tools).
    """
    classification = state["ticket_state"].classification

    sys_prompt = AGENT_SYSTEM_PROMPT.format(
        category=classification.category.value,
        urgency=classification.urgency.value,
        confidence=classification.confidence,
    )

    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = await agent_llm.ainvoke(messages)

    return {"messages": [response]}