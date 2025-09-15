# src/config.py

import os
from typing import Dict
from dotenv import load_dotenv

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()

DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

# --- Azure Credentials ---
AZURE_DI_KEY = os.getenv("AZURE_FORMRECOGNIZER_KEY")
AZURE_DI_ENDPOINT = os.getenv("AZURE_FORMRECOGNIZER_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# --- Settings ---
WRITE_CHUNK_FILES = False
MAX_RETRIES_LIBRARIAN = 0
MIN_HEADINGS_PER_CHUNK = 2
MAX_HEADINGS_PER_CHUNK = 32
CONTEXT_K = 6

# --- Model Config ---
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Path Generation ---
def get_paths(pdf_base_name: str) -> Dict[str, str]:
    out_dir = os.path.join(PROJECT_ROOT, "output")
    return {
        "pdf_input": os.path.join(PROJECT_ROOT, "input", f"{pdf_base_name}.pdf"),
        "output_dir": out_dir,
        "raw_md": os.path.join(out_dir, f"{pdf_base_name}_raw.md"),
        "refined_md": os.path.join(out_dir, f"{pdf_base_name}_refined.md"),
        "chunk_plan": os.path.join(out_dir, f"{pdf_base_name}_chunk_plan.json"),
        "chunks_dir": os.path.join(out_dir, f"{pdf_base_name}_chunks"),
        "vector_store": os.path.join(out_dir, f"{pdf_base_name}_vector_store.faiss"),
        "metadata": os.path.join(out_dir, f"{pdf_base_name}_metadata.json"),
        "bm25_pickle": os.path.join(out_dir, f"{pdf_base_name}_bm25.pkl"),
    }

# --- Credential Check ---
def check_credentials() -> bool:
    print("Checking Azure credentials...")
    ok = True
    if not AZURE_DI_KEY or not AZURE_DI_ENDPOINT:
        print("  Missing Document Intelligence credentials.")
        ok = False
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        print("  Missing Azure OpenAI credentials.")
        ok = False
    if not ok:
        print(f"  .env file checked at: {DOTENV_PATH}")
    return ok

if __name__ == "__main__":
    check_credentials()
    paths = get_paths("Sample")
    print("\nGenerated paths:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
