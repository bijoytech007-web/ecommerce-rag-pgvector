import os
import json
import time
from decimal import Decimal
import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, ClientError

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Generation models with fallback order
PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


def get_query_embedding(query_text: str):
    """Turns the query into a 768-dim vector matching Neon DB schema."""
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query_text,
        config={"output_dimensionality": 768},
    )
    return response.embeddings[0].values


def vector_search(query_text: str, limit: int = 2):
    """Executes pgvector cosine similarity search (<=>) on Neon."""
    query_vector = get_query_embedding(query_text)
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, sku, name, category, description, base_price, stock_qty,
                   (embedding <=> %s::vector) AS distance
            FROM products
            ORDER BY distance ASC
            LIMIT %s;
        """,
            (query_vector, limit),
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    return [
        {
            "id": r[0],
            "sku": r[1],
            "name": r[2],
            "category": r[3],
            "description": r[4],
            "base_price": float(r[5]),
            "stock_qty": r[6],
            "similarity_score": round(1 - float(r[7]), 4),
        }
        for r in rows
    ]


def calculate_tier_price(product_id: int, quantity: int):
    """Calculates deterministic wholesale tiered discount from database slabs."""
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT name, base_price, stock_qty FROM products WHERE id = %s",
            (product_id,),
        )
        prod = cur.fetchone()
        name = prod[0]
        base_price = Decimal(str(prod[1]))
        stock = prod[2]

        cur.execute(
            """
            SELECT discount_percentage 
            FROM product_pricing_slabs
            WHERE product_id = %s 
              AND min_qty <= %s 
              AND (max_qty IS NULL OR max_qty >= %s)
            ORDER BY min_qty DESC
            LIMIT 1;
        """,
            (product_id, quantity, quantity),
        )

        slab = cur.fetchone()
        discount = Decimal(str(slab[0])) if slab else Decimal("0.00")

        unit_price = base_price * (
            Decimal("1.00") - (discount / Decimal("100.00"))
        )
        total = unit_price * quantity
    conn.close()

    return {
        "product_name": name,
        "ordered_qty": quantity,
        "is_in_stock": stock >= quantity,
        "stock_available": stock,
        "base_unit_price": float(base_price),
        "discount_applied": f"{float(discount)}%",
        "final_unit_price": float(round(unit_price, 2)),
        "total_amount": float(round(total, 2)),
    }


def call_llm_with_resilience(prompt: str, max_retries: int = 3):
    """Handles transient 503 errors with exponential backoff and model fallbacks."""
    for model_name in PREFERRED_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                return response.text
            except ServerError as e:
                # 503 high demand spike handler
                wait_time = 2 * (attempt + 1)
                print(
                    f"Warning: {model_name} busy (503). Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            except ClientError as e:
                print(
                    f"Notice: Model {model_name} unavailable, routing to next fallback..."
                )
                break  # try next model in PREFERRED_MODELS
    raise RuntimeError("All configured LLM models were temporarily unavailable.")


def process_customer_order(query: str, quantity: int = 1):
    matches = vector_search(query, limit=1)
    if not matches:
        return "No matching products found in the catalog."

    best_match = matches[0]
    pricing = calculate_tier_price(best_match["id"], quantity)

    prompt = f"""You are a professional B2B Wholesale Sales Assistant.
Using the verified database product match and slab-pricing calculation, generate a professional quote.

DATABASE CATALOG CONTEXT:
{json.dumps(best_match, indent=2)}

DATABASE PRICING CALCULATION:
{json.dumps(pricing, indent=2)}

CUSTOMER REQUEST:
Query: "{query}"
Requested Quantity: {quantity}

REQUIREMENTS:
1. Provide a clear, professional summary with SKU, product name, inventory stock confirmation, unit base price, applied volume discount, and total quote.
2. If the requested quantity exceeds stock, explicitly note the backorder status.
"""

    return call_llm_with_resilience(prompt)


if __name__ == "__main__":
    print("=" * 60)
    print("  Enterprise RAG Sales Assistant (Neon pgvector + Gemini)  ")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Describe what you are looking for: ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            break
        if not user_query:
            continue

        raw_qty = input("Quantity needed (default 1): ").strip()
        try:
            qty = int(raw_qty) if raw_qty else 1
        except ValueError:
            print("Invalid quantity. Using default of 1.")
            qty = 1

        print("\nSearching PostgreSQL vector catalog & generating quote...")
        try:
            result = process_customer_order(user_query, quantity=qty)
            print("\n--- QUOTE RESPONSE ---")
            print(result)
            print("-" * 60 + "\n")
        except Exception as err:
            print(f"Error: {err}\n")