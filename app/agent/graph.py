# app/agent/graph.py

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, ToolMessage
from app.models.schemas import AgentState
from app.agent.nodes import triage_ticket, resolve_ticket, tool_node

_TERMINAL_TOOLS = frozenset({"send_reply", "escalate"})


def _terminal_tool_was_called(state: AgentState) -> bool:
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "") or ""
            if tool_name in _TERMINAL_TOOLS:
                return True
    return False


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    if not _terminal_tool_was_called(state):
        return "enforce_terminal"
    return END


def route_after_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        tool_name = getattr(last, "name", None) or ""
        if tool_name in _TERMINAL_TOOLS:
            return END
    return "agent"


async def enforce_terminal_node(state: AgentState) -> dict:
    # Import here to avoid circular import at module load time
    from app.tools.write_tools import send_reply, escalate

    ticket_id = state["ticket_state"].ticket_id

    last_ai_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_text = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    if last_ai_text and len(last_ai_text.strip()) >= 30:
        result = await send_reply.ainvoke({
            "ticket_id": ticket_id,
            "message": last_ai_text.strip(),
        })
        tool_name = "send_reply"
    else:
        result = await escalate.ainvoke({
            "ticket_id": ticket_id,
            "issue_summary": "Agent completed reasoning but did not send a reply.",
            "recommended_path": "Review agent reasoning and send appropriate customer response.",
            "priority": "low",
        })
        tool_name = "escalate"

    synthetic = ToolMessage(
        content=str(result),
        tool_call_id="safety_net_enforcement",
        name=tool_name,
    )
    return {"messages": [synthetic]}


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("triage",           triage_ticket)
    builder.add_node("agent",            resolve_ticket)
    builder.add_node("tools",            tool_node)
    builder.add_node("enforce_terminal", enforce_terminal_node)

    builder.add_edge(START,    "triage")
    builder.add_edge("triage", "agent")
    builder.add_edge("enforce_terminal", END)

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools":            "tools",
            END:                END,
            "enforce_terminal": "enforce_terminal",
        },
    )
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", END: END},
    )

    return builder.compile()


agent_graph = build_graph()