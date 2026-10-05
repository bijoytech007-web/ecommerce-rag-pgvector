import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available embedding models for your key:")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(f" -> {m.name}")