import json
import os
 
_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
 
def _load(filename: str) -> list:
    with open(os.path.join(_DATA_DIR, filename)) as f:
        return json.load(f)
 
CUSTOMERS_DB: list[dict] = _load("customers.json")
ORDERS_DB:    list[dict] = _load("orders.json")
PRODUCTS_DB:  list[dict] = _load("products.json")