import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def init_db():
    print("Connecting to Neon PostgreSQL...")
    conn = psycopg.connect(DB_URL)
    with conn.cursor() as cur:
        # Table 1: Products with 768-dimension vectors for Gemini text-embedding-004
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                sku VARCHAR(64) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(100),
                description TEXT,
                base_price NUMERIC(10, 2) NOT NULL,
                stock_qty INT DEFAULT 0,
                embedding vector(768)
            );
        """)

        # Table 2: Slab volume discounts (e.g., 10-49 units: 10% off)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_pricing_slabs (
                id SERIAL PRIMARY KEY,
                product_id INT REFERENCES products(id) ON DELETE CASCADE,
                min_qty INT NOT NULL,
                max_qty INT,
                discount_percentage NUMERIC(5, 2) DEFAULT 0.00
            );
        """)

        conn.commit()
    conn.close()
    print("Success: Tables created in Neon PostgreSQL!")

if __name__ == "__main__":
    init_db()