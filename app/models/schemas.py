from pydantic import BaseModel
from typing import Optional, List, Annotated, TypedDict
from enum import Enum
from langchain_core.messages import AnyMessage


class UrgencyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class TicketCategory(str, Enum):
    refund = "refund"
    return_ = "return"
    cancel = "cancel"
    warranty = "warranty"
    exchange = "exchange"
    info = "info"


class ResolutionStatus(str, Enum):
    resolved = "resolved"
    escalated = "escalated"
    failed = "failed"
    pending = "pending"


class ToolCall(BaseModel):
    tool_name: str
    input: dict
    output: str
    reasoning: str

class TicketClassification(BaseModel):
    category: TicketCategory
    urgency: UrgencyLevel
    resolvable: bool
    confidence: float
    reasoning: str


class EscalationSummary(BaseModel):
    issue: str
    what_was_tried: str
    recommended_path: str
    priority: UrgencyLevel


class TicketResolution(BaseModel):
    ticket_id: str
    customer_email: str = "unknown" 
    status: ResolutionStatus = ResolutionStatus.pending
    classification: Optional[TicketClassification] = None 
    tool_calls: List[ToolCall] = []
    customer_reply: Optional[str] = None
    escalation: Optional[EscalationSummary] = None
    error: Optional[str] = None

TicketState = TicketResolution


def add_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    return left + right


class AgentState(TypedDict):
    """The memory object passed between LangGraph nodes."""
    messages: Annotated[list[AnyMessage], add_messages]
    ticket_state: TicketState