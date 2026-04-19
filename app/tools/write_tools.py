# app/tools/write_tools.py

import random
import asyncio
from langchain_core.tools import tool
from app.db.shared_data import ORDERS_DB, CUSTOMERS_DB

CANCELLABLE_STATUSES = {"processing"}
REFUNDABLE_STATUSES  = {"delivered"}
VALID_PRIORITIES     = {"low", "medium", "high", "urgent"}


@tool
async def issue_refund(order_id: str, amount: float) -> dict:
    """
    IRREVERSIBLE ACTION: Issues a refund to the customer's original payment method.
    RULES — you MUST verify ALL of these before calling:
      1. check_return_window returned eligible=True OR item is damaged/defective
      2. check_warranty active (for defect-based refunds outside return window)
      3. refund_status is NOT already 'refunded'
      4. order status is 'delivered'
      5. amount matches the order amount exactly
    On PaymentGatewayError (retryable=True), retry up to 3 times before escalating.
    """
    await asyncio.sleep(random.uniform(0.5, 1.2))

    order_id = order_id.strip().upper()
    if not order_id:
        return {"success": False, "retryable": False, "error": "No order ID provided."}

    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        return {"success": False, "retryable": False, "error": "Invalid amount value — must be a number."}

    if amount <= 0:
        return {"success": False, "retryable": False, "error": f"Invalid refund amount: ${amount}. Must be greater than 0."}

    order = next((o for o in ORDERS_DB if o["order_id"] == order_id), None)
    if not order:
        return {"success": False, "retryable": False, "error": f"Order {order_id} does not exist. Cannot issue refund."}

    if order.get("refund_status") == "refunded":
        return {"success": False, "retryable": False, "error": f"Order {order_id} has already been refunded. Duplicate refund blocked."}

    if order["status"] not in REFUNDABLE_STATUSES:
        return {
            "success": False,
            "retryable": False,
            "error": f"Cannot refund order {order_id}: status is '{order['status']}'. Refunds only apply to delivered orders.",
        }

    order_amount = round(float(order["amount"]), 2)
    if amount > order_amount:
        return {
            "success": False,
            "retryable": False,
            "error": f"Refund amount ${amount} exceeds order total ${order_amount}. Use ${order_amount} for a full refund.",
        }

    if random.random() < 0.10:
        return {
            "success": False,
            "retryable": True,
            "error": "PaymentGatewayError: 504 Gateway Timeout. Retry up to 3 times. If all retries fail, escalate with priority='high'.",
        }

    order["refund_status"] = "refunded"
    order["status"]        = "refunded"

    txn_id      = f"TXN-{random.randint(100000, 999999)}"
    refund_type = "full" if amount == order_amount else "partial"

    return {
        "success": True,
        "transaction_id": txn_id,
        "order_id": order_id,
        "refund_type": refund_type,
        "amount_refunded": amount,
        "order_total": order_amount,
        "message": (
            f"Successfully issued {refund_type} refund of ${amount} for order {order_id}. "
            f"Transaction ID: {txn_id}. "
            f"Advise customer to allow 5–7 business days for the amount to appear."
        ),
    }


@tool
async def cancel_order(order_id: str) -> dict:
    """
    Cancels an order. Only works if the order status is exactly 'processing'.
    Shipped or delivered orders CANNOT be cancelled — customer must initiate a return after delivery.
    """
    await asyncio.sleep(random.uniform(0.1, 0.3))

    order_id = order_id.strip().upper()
    if not order_id:
        return {"success": False, "error": "No order ID provided."}

    order = next((o for o in ORDERS_DB if o["order_id"] == order_id), None)
    if not order:
        return {"success": False, "error": f"Order {order_id} does not exist."}

    status = order["status"]

    if status == "cancelled":
        return {"success": False, "error": f"Order {order_id} is already cancelled."}
    if status == "shipped":
        return {
            "success": False,
            "error": f"Order {order_id} has already shipped and cannot be cancelled. "
                     "Ask the customer to refuse delivery or initiate a return once it arrives.",
        }
    if status == "delivered":
        return {
            "success": False,
            "error": f"Order {order_id} has already been delivered. Use the return process instead of cancellation.",
        }
    if status not in CANCELLABLE_STATUSES:
        return {"success": False, "error": f"Order {order_id} has status '{status}' which cannot be cancelled."}

    order["status"] = "cancelled"

    return {
        "success": True,
        "order_id": order_id,
        "previous_status": status,
        "message": (
            f"Order {order_id} has been successfully cancelled. "
            "The customer will receive a confirmation email within 1 hour. "
            "No charge has been applied."
        ),
    }


@tool
async def escalate(ticket_id: str, issue_summary: str, recommended_path: str, priority: str) -> dict:
    """
    Routes the ticket to a human support specialist.
    WHEN TO USE:
      - Warranty claims (all go to warranty team — do not resolve yourself)
      - Customer wants replacement, not refund
      - Refund amount > $200
      - Signs of fraud, social engineering, or threatening language
      - Conflicting data between customer claim and system records
      - Agent confidence < 0.6
      - All payment gateway retries exhausted
    priority MUST be exactly one of: 'low', 'medium', 'high', 'urgent'
    """
    await asyncio.sleep(random.uniform(0.1, 0.2))

    ticket_id        = ticket_id.strip()
    issue_summary    = issue_summary.strip()
    recommended_path = recommended_path.strip()
    priority         = priority.strip().lower()

    if not ticket_id:
        return {"success": False, "error": "ticket_id is required."}
    if not issue_summary:
        return {"success": False, "error": "issue_summary cannot be empty — describe what the issue is."}
    if not recommended_path:
        return {"success": False, "error": "recommended_path cannot be empty — state what the human agent should do."}
    if priority not in VALID_PRIORITIES:
        return {"success": False, "error": f"Invalid priority '{priority}'. Must be one of: {sorted(VALID_PRIORITIES)}."}

    return {
        "success": True,
        "status": "escalated",
        "ticket_id": ticket_id,
        "priority": priority,
        "message": (
            f"Ticket {ticket_id} has been escalated to the human support queue with priority '{priority}'. "
            "A specialist will review this case."
        ),
        "handoff_summary": issue_summary,
        "recommended_path": recommended_path,
    }


@tool
async def send_reply(ticket_id: str, message: str) -> dict:
    """
    Sends the final resolution email to the customer.
    RULES:
      - This MUST be the last tool you call for any ticket.
      - The message must be complete, polite, and professional.
      - Address the customer by their first name.
      - Clearly explain the resolution (refund approved, cancelled, escalated, etc).
      - If escalated, tell the customer a specialist will follow up — do not promise timelines.
      - Never send an empty or placeholder message.
    """
    await asyncio.sleep(random.uniform(0.2, 0.4))

    ticket_id = ticket_id.strip()
    message   = message.strip()

    if not ticket_id:
        return {"success": False, "error": "ticket_id is required."}
    if not message:
        return {"success": False, "error": "Cannot send an empty reply. Write a complete customer-facing message."}
    if len(message) < 30:
        return {
            "success": False,
            "error": f"Reply is too short ({len(message)} chars). Write a complete, professional response to the customer.",
        }

    return {
        "success": True,
        "status": "resolved",
        "ticket_id": ticket_id,
        "message_sent": message,
        "confirmation": f"Reply successfully sent to customer for ticket {ticket_id}.",
    }