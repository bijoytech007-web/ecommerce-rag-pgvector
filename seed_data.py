import os
import psycopg
from dotenv import load_dotenv
from google import genai

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize official google-genai client
client = genai.Client(api_key=GOOGLE_API_KEY)

SAMPLE_PRODUCTS = [
    {
        "sku": "BOX-HVY-01",
        "name": "Heavy-Duty Corrugated Shipping Boxes (Pack of 25)",
        "category": "Packaging",
        "description": "Double-wall corrugated shipping boxes engineered for heavy industrial goods and fragile items.",
        "base_price": 50.00,
        "stock_qty": 500,
        "slabs": [
            {"min_qty": 1, "max_qty": 9, "discount": 0.00},
            {"min_qty": 10, "max_qty": 49, "discount": 10.00},
            {"min_qty": 50, "max_qty": None, "discount": 20.00}
        ]
    },
    {
        "sku": "CHAIR-ERG-02",
        "name": "Ergonomic Lumbar Executive Task Chair",
        "category": "Office Furniture",
        "description": "High-back breathable mesh chair with 3D armrests and active lumbar support for long working sessions.",
        "base_price": 190.00,
        "stock_qty": 120,
        "slabs": [
            {"min_qty": 1, "max_qty": 4, "discount": 0.00},
            {"min_qty": 5, "max_qty": 19, "discount": 8.00},
            {"min_qty": 20, "max_qty": None, "discount": 15.00}
        ]
    },
    {
        "sku": "TAPE-IND-03",
        "name": "Industrial Waterproof Carton Sealing Tape (6 Rolls)",
        "category": "Packaging",
        "description": "Ultra-adhesive heavy-duty packaging tape resistant to moisture, extreme temperatures, and carton splitting.",
        "base_price": 28.00,
        "stock_qty": 300,
        "slabs": [
            {"min_qty": 1, "max_qty": 9, "discount": 0.00},
            {"min_qty": 10, "max_qty": 29, "discount": 5.00},
            {"min_qty": 30, "max_qty": None, "discount": 12.50}
        ]
    }
]

def get_embedding(text: str):
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768},
    )
    return response.embeddings[0].values
def seed():
    print("Connecting to Neon PostgreSQL and generating embeddings...")
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        for item in SAMPLE_PRODUCTS:
            embed_text = f"{item['name']} | {item['category']} | {item['description']}"
            print(f"Embedding SKU: {item['sku']}...")
            vector = get_embedding(embed_text)

            cur.execute("""
                INSERT INTO products (sku, name, category, description, base_price, stock_qty, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sku) DO UPDATE SET 
                    base_price = EXCLUDED.base_price,
                    stock_qty = EXCLUDED.stock_qty,
                    embedding = EXCLUDED.embedding
                RETURNING id;
            """, (item["sku"], item["name"], item["category"], item["description"], item["base_price"], item["stock_qty"], vector))
            
            product_id = cur.fetchone()[0]

            cur.execute("DELETE FROM product_pricing_slabs WHERE product_id = %s", (product_id,))
            for slab in item["slabs"]:
                cur.execute("""
                    INSERT INTO product_pricing_slabs (product_id, min_qty, max_qty, discount_percentage)
                    VALUES (%s, %s, %s, %s)
                """, (product_id, slab["min_qty"], slab["max_qty"], slab["discount"]))

        conn.commit()
    conn.close()
    print("Database seeding completed successfully.")

if __name__ == "__main__":
    seed()