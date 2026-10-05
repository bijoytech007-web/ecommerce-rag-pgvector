# Enterprise B2B E-Commerce RAG & Dynamic Pricing Engine

A production-grade Retrieval-Augmented Generation (RAG) sales pipeline combining **PostgreSQL (`pgvector`)**, **Google Gemini**, and **Streamlit**. 

Unlike conventional flat-file or in-memory vector demonstrations, this system uses a managed cloud PostgreSQL instance to execute real-time cosine distance vector searches across an enterprise catalog, couples it with deterministic database-driven wholesale pricing slabs, and returns a validated quote with automated fallback handling.

---

## Architecture Overview

```text
[ Customer Query & Order Quantity ]
               │
               ▼
   [ Gemini Embedding Engine ] ──> (768-dim Query Vector)
                                             │
                                             ▼
                 [ Neon PostgreSQL (pgvector + HNSW Index) ]
                                             │
                                ┌────────────┴────────────┐
                                ▼                         ▼
                    [ Semantic Catalog Match ]   [ Tier Pricing Slabs ]
                                │                         │
                                └────────────┬────────────┘
                                             ▼
                            [ Business Logic & Stock Check ]
                                             │
                                             ▼
                            [ Resilient LLM Quote Generator ]
                                (Gemini Flash with Fallback)
                                             │
                                             ▼
                                [ Streamlit Web Dashboard ]