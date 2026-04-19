# app/tools/read_tools.py

import random
import asyncio
from datetime import datetime
from langchain_core.tools import tool
from app.db.shared_data import CUSTOMERS_DB, ORDERS_DB, PRODUCTS_DB



@tool
async def get_customer(email: str) -> dict:
    """
    Look up a customer by email and return their full profile, tier, and notes.
    Always call this first whenever a customer email is present in the ticket.
    Returns tier (standard / premium / VIP) which affects refund policy exceptions.
    """
    await asyncio.sleep(random.uniform(0.1, 0.2))

    if random.random() < 0.10:
        return {
            "success": False,
            "error": "MalformedResponse: Data corrupted during transit. Please retry.",
        }

    try:
        for customer in CUSTOMERS_DB:
            if customer["email"].lower() == email.lower():
                return {"success": True, "customer": customer}
        return {"success": False, "error": f"No customer found with email: {email}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
async def get_order(order_id: str) -> dict:
    """
    Look up an order by order ID.
    Returns order status, delivery date, product ID, amount, and refund status.
    Use this before any refund, cancel, or return action.
    """
    await asyncio.sleep(random.uniform(0.1, 0.3))
    if random.random() < 0.15:
        return {
            "success": False,
            "error": "TimeoutError: Database failed to respond. You must retry this tool call.",
        }

    try:
        for order in ORDERS_DB:
            if order["order_id"] == order_id:
                return {"success": True, "order": order}
        return {"success": False, "error": f"Order not found: {order_id}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
async def get_product(product_id: str) -> dict:
    """
    Look up a product by product ID.
    Returns product name, category, warranty months, return window, and returnable flag.
    """
    try:
        for product in PRODUCTS_DB:
            if product["product_id"] == product_id:
                return {"success": True, "product": product}
        return {"success": False, "error": f"Product not found: {product_id}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
async def get_orders_by_customer_email(email: str) -> dict:
    """
    Find ALL orders for a customer using their email address.
    Use this when no order ID is present in the ticket — it lets you identify
    which order the customer is referring to.
    """
    await asyncio.sleep(random.uniform(0.1, 0.2))

    try:
        customer = next(
            (c for c in CUSTOMERS_DB if c["email"].lower() == email.lower()), None
        )
        if not customer:
            return {"success": False, "error": f"No customer found with email: {email}"}

        customer_id     = customer["customer_id"]
        customer_orders = [o for o in ORDERS_DB if o["customer_id"] == customer_id]

        if not customer_orders:
            return {"success": False, "error": f"No orders found for customer: {email}"}

        return {"success": True, "orders": customer_orders}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
async def check_return_window(order_id: str) -> dict:
    """
    Check if an order is still within the return window.
    Evaluates return_deadline, refund status, product category, and online
    registration policies (registered devices cannot be returned).

    Returns:
      - within_window: bool
      - return_deadline: date string
      - days_since_delivery: int
      - registered_online: bool (if True, item is non-returnable)
      - returnable: bool (product-level flag)
    """
    try:
        order = next((o for o in ORDERS_DB if o["order_id"] == order_id), None)
        if not order:
            return {"success": False, "error": f"Order not found: {order_id}"}

        if order.get("refund_status") == "refunded":
            return {
                "success":       True,
                "order_id":      order_id,
                "within_window": False,
                "reason":        "Refund already processed for this order",
                "refund_status": "refunded",
            }

        if not order.get("delivery_date"):
            return {
                "success":       True,
                "order_id":      order_id,
                "within_window": False,
                "reason": (
                    f"Order has not been delivered yet. "
                    f"Current status: {order['status']}"
                ),
            }

        product = next(
            (p for p in PRODUCTS_DB if p["product_id"] == order["product_id"]), None
        )
        if not product:
            return {"success": False, "error": f"Product not found for order: {order_id}"}

        return_deadline     = datetime.strptime(order["return_deadline"], "%Y-%m-%d")
        today               = datetime.today()
        within_window       = today <= return_deadline
        days_since_delivery = (
            today - datetime.strptime(order["delivery_date"], "%Y-%m-%d")
        ).days
        registered_online   = "registered online" in order.get("notes", "").lower()

        return {
            "success":             True,
            "order_id":            order_id,
            "product_name":        product["name"],
            "product_category":    product["category"],
            "delivery_date":       order["delivery_date"],
            "return_deadline":     order["return_deadline"],
            "return_window_days":  product["return_window_days"],
            "days_since_delivery": days_since_delivery,
            "within_window":       within_window,
            "registered_online":   registered_online,
            "returnable":          product["returnable"],
            "product_notes":       product.get("notes", ""),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
async def check_refund_eligibility(order_id: str) -> dict:
    """
    Check if an order is eligible for a refund.
    Evaluates the return window, refund status, product category rules, and
    online registration policies.

    This is the primary tool to call before issuing any refund.
    It returns a clear eligible: True/False with a reason explaining the decision.

    Returns:
      - eligible: bool — True means you can proceed to issue_refund
      - within_window: bool
      - return_deadline: date string
      - days_since_delivery: int
      - registered_online: bool
      - returnable: bool
      - reason: str (human-readable explanation of eligibility decision)
    """
    try:
        order = next((o for o in ORDERS_DB if o["order_id"] == order_id), None)
        if not order:
            return {
                "success":  False,
                "eligible": False,
                "error":    f"Order not found: {order_id}",
            }

        if order.get("refund_status") == "refunded":
            return {
                "success":       True,
                "eligible":      False,
                "order_id":      order_id,
                "within_window": False,
                "reason":        "Order has already been refunded. Duplicate refund blocked.",
                "refund_status": "refunded",
            }

        if not order.get("delivery_date"):
            return {
                "success":       True,
                "eligible":      False,
                "order_id":      order_id,
                "within_window": False,
                "reason": (
                    f"Order has not been delivered yet. "
                    f"Current status: {order['status']}. "
                    "Refunds only apply to delivered orders."
                ),
            }

        product = next(
            (p for p in PRODUCTS_DB if p["product_id"] == order["product_id"]), None
        )
        if not product:
            return {
                "success":  False,
                "eligible": False,
                "error":    f"Product not found for order: {order_id}",
            }

        return_deadline     = datetime.strptime(order["return_deadline"], "%Y-%m-%d")
        today               = datetime.today()
        within_window       = today <= return_deadline
        days_since_delivery = (
            today - datetime.strptime(order["delivery_date"], "%Y-%m-%d")
        ).days
        registered_online   = "registered online" in order.get("notes", "").lower()
        returnable          = product.get("returnable", True)
        
        if not returnable:
            eligible = False
            reason   = (
                f"Product '{product['name']}' is marked non-returnable "
                "(activated/registered device or final sale item)."
            )
        elif registered_online:
            eligible = False
            reason   = (
                f"Order {order_id} cannot be returned — the device was registered "
                "online after purchase. Registered devices are non-returnable per policy."
            )
        elif not within_window:
            eligible = False
            reason   = (
                f"Return window expired. Delivery was {order['delivery_date']}, "
                f"return deadline was {order['return_deadline']} "
                f"({product['return_window_days']}-day window). "
                f"{days_since_delivery} days have passed since delivery."
            )
        else:
            eligible = True
            reason   = (
                f"Order is within the {product['return_window_days']}-day return window. "
                f"Delivered {order['delivery_date']}, deadline {order['return_deadline']}. "
                f"{days_since_delivery} days since delivery. Refund is eligible."
            )

        return {
            "success":             True,
            "eligible":            eligible,
            "order_id":            order_id,
            "product_name":        product["name"],
            "product_category":    product["category"],
            "delivery_date":       order["delivery_date"],
            "return_deadline":     order["return_deadline"],
            "return_window_days":  product["return_window_days"],
            "days_since_delivery": days_since_delivery,
            "within_window":       within_window,
            "registered_online":   registered_online,
            "returnable":          returnable,
            "reason":              reason,
        }
    except Exception as e:
        return {"success": False, "eligible": False, "error": str(e)}


@tool
async def check_warranty(order_id: str) -> dict:
    """
    Check if an order is within the warranty period based on delivery date.
    Warranty covers manufacturing defects only — not physical damage or misuse.

    Returns:
      - warranty_active: bool
      - warranty_expiry: date string
      - days_remaining: int
      - warranty_months: int (0 = no warranty for this product)
    """
    try:
        order = next((o for o in ORDERS_DB if o["order_id"] == order_id), None)
        if not order:
            return {"success": False, "error": f"Order not found: {order_id}"}

        if not order.get("delivery_date"):
            return {"success": False, "error": "Order has not been delivered yet"}

        product = next(
            (p for p in PRODUCTS_DB if p["product_id"] == order["product_id"]), None
        )
        if not product:
            return {"success": False, "error": f"Product not found for order: {order_id}"}

        warranty_months = product.get("warranty_months", 0)
        if warranty_months == 0:
            return {
                "success":         True,
                "order_id":        order_id,
                "product_name":    product["name"],
                "warranty_active": False,
                "reason":          "This product has no warranty coverage.",
            }

        delivery_date   = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
        warranty_expiry = delivery_date.replace(
            month=(
                (delivery_date.month - 1 + warranty_months) % 12
            ) + 1,
            year=delivery_date.year
            + (delivery_date.month - 1 + warranty_months) // 12,
        )
        today           = datetime.today()
        warranty_active = today <= warranty_expiry

        return {
            "success":          True,
            "order_id":         order_id,
            "product_name":     product["name"],
            "product_category": product["category"],
            "delivery_date":    order["delivery_date"],
            "warranty_months":  warranty_months,
            "warranty_expiry":  warranty_expiry.strftime("%Y-%m-%d"),
            "warranty_active":  warranty_active,
            "days_remaining":   (warranty_expiry - today).days if warranty_active else 0,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}