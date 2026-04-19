from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "shopwave")

# FIX: Use Async motor client to prevent event loop blocking
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]

tickets_collection = db["tickets"]
audit_collection = db["audit_logs"]


async def save_resolution(resolution: dict):
    """Save or update a ticket resolution in MongoDB."""
    await tickets_collection.update_one(
        {"ticket_id": resolution.get("ticket_id")},
        {"$set": resolution},
        upsert=True
    )


async def save_audit_log(audit: dict):
    """Append an audit log entry to MongoDB."""
    # FIX: Use update_one with upsert to prevent duplicates on retries
    await audit_collection.update_one(
        {"ticket_id": audit.get("ticket_id")},
        {"$set": audit},
        upsert=True
    )


async def get_all_resolutions():
    """Fetch all ticket resolutions."""
    cursor = tickets_collection.find({}, {"_id": 0})
    return await cursor.to_list(length=1000)


async def get_resolution_by_id(ticket_id: str):
    """Fetch a single ticket resolution by ID."""
    return await tickets_collection.find_one({"ticket_id": ticket_id}, {"_id": 0})


async def get_all_audit_logs():
    """Fetch all audit logs."""
    cursor = audit_collection.find({}, {"_id": 0})
    return await cursor.to_list(length=1000)