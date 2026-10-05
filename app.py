import os
import json
import time
from decimal import Decimal
import psycopg
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai.errors import ServerError, ClientError

# Page Configuration
st.set_page_config(
    page_title="AI B2B Wholesale Portal",
    page_icon="📦",
    layout="wide"
)

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


# ------------------ Database & Embeddings ------------------
def get_query_embedding(query_text: str):
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query_text,
        config={"output_dimensionality": 768},
    )
    return response.embeddings[0].values


def vector_search(query_text: str, limit: int = 1):
    query_vector = get_query_embedding(query_text)
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, sku, name, category, description, base_price, stock_qty,
                   (embedding <=> %s::vector) AS distance
            FROM products
            ORDER BY distance ASC
            LIMIT %s;
        """, (query_vector, limit))
        row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "sku": row[1],
        "name": row[2],
        "category": row[3],
        "description": row[4],
        "base_price": float(row[5]),
        "stock_qty": row[6],
        "match_confidence": f"{round((1 - float(row[7])) * 100, 1)}%"
    }


def calculate_tier_price(product_id: int, quantity: int):
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT name, base_price, stock_qty FROM products WHERE id = %s", (product_id,))
        prod = cur.fetchone()
        name = prod[0]
        base_price = Decimal(str(prod[1]))
        stock = prod[2]

        cur.execute("""
            SELECT discount_percentage 
            FROM product_pricing_slabs
            WHERE product_id = %s 
              AND min_qty <= %s 
              AND (max_qty IS NULL OR max_qty >= %s)
            ORDER BY min_qty DESC
            LIMIT 1;
        """, (product_id, quantity, quantity))
        
        slab = cur.fetchone()
        discount = Decimal(str(slab[0])) if slab else Decimal("0.00")

        unit_price = base_price * (Decimal("1.00") - (discount / Decimal("100.00")))
        total = unit_price * quantity
    conn.close()

    return {
        "product_name": name,
        "ordered_qty": quantity,
        "is_in_stock": stock >= quantity,
        "stock_available": stock,
        "base_unit_price": float(base_price),
        "discount_applied": float(discount),
        "final_unit_price": float(round(unit_price, 2)),
        "total_amount": float(round(total, 2))
    }


def get_all_products():
    """Fetches full inventory to show in the sidebar."""
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT sku, name, base_price, stock_qty FROM products ORDER BY id ASC;")
        rows = cur.fetchall()
    conn.close()
    return [{"sku": r[0], "name": r[1], "price": float(r[2]), "stock": r[3]} for r in rows]


def call_llm(prompt: str, max_retries: int = 3):
    for model_name in PREFERRED_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except ServerError:
                time.sleep(2 * (attempt + 1))
            except ClientError:
                break
    return "Sales assistant temporarily unavailable due to server demand."


# ------------------ Streamlit UI ------------------
st.title("📦 B2B Wholesale Portal & AI Quoting Engine")
st.caption("Powered by Neon Serverless PostgreSQL (`pgvector`) & Gemini")

# Sidebar: Live Catalog
with st.sidebar:
    st.subheader("⚡ Live Catalog Status")
    st.success("Connected to Neon PostgreSQL")
    catalog_items = get_all_products()
    st.dataframe(catalog_items, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown("**Wholesale Slabs:**")
    st.caption("• 1–9 units: Standard Price\n• 10–49 units: 10% Off\n• 50+ units: 20% Off")

col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input(
        "Search catalog by semantic intent (no exact keywords needed):",
        placeholder="e.g., Heavy duty shipping boxes for fragile equipment"
    )
with col2:
    order_qty = st.number_input("Order Quantity", min_value=1, max_value=5000, value=60, step=1)

if st.button("Generate Commercial Quote", type="primary", use_container_width=True):
    if not search_query.strip():
        st.warning("Please enter a product description or search query.")
    else:
        with st.spinner("Executing pgvector cosine search and calculating slabs..."):
            matched_product = vector_search(search_query)

        if not matched_product:
            st.error("No relevant items found in catalog.")
        else:
            quote = calculate_tier_price(matched_product["id"], order_qty)

            # Metrics Row
            st.subheader(f"Matched: {matched_product['name']} (`{matched_product['sku']}`)")
            st.write(matched_product['description'])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Base Price", f"${quote['base_unit_price']:.2f}")
            m2.metric("Applied Discount", f"{quote['discount_applied']}%", delta=f"-{quote['discount_applied']}%" if quote['discount_applied'] > 0 else None)
            m3.metric("Final Unit Price", f"${quote['final_unit_price']:.2f}")
            m4.metric("Total Quote", f"${quote['total_amount']:.2f}")

            if not quote["is_in_stock"]:
                st.warning(f"⚠️ Requested {order_qty} units, but only {quote['stock_available']} are currently in stock (Backorder applies).")
            else:
                st.success(f"✅ In Stock: {quote['stock_available']} units available.")

            # LLM Grounded Pitch
            st.markdown("---")
            st.subheader("🤖 AI Sales Representative Statement")

            llm_prompt = f"""You are a professional B2B Wholesale Sales Assistant.
Using the verified database product match and slab-pricing calculation, generate a professional quote.

DATABASE CATALOG CONTEXT:
{json.dumps(matched_product, indent=2)}

DATABASE PRICING CALCULATION:
{json.dumps(quote, indent=2)}

CUSTOMER REQUEST:
Query: "{search_query}"
Requested Quantity: {order_qty}

Generate a concise, courteous business response confirming the order total, discount tier, and shipping fulfillment status."""

            with st.spinner("Generating AI response..."):
                sales_pitch = call_llm(llm_prompt)
                st.markdown(sales_pitch)