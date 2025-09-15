# src/rag_logic.py
# (all your core logic, same behavior, wrapped for import by Streamlit UI)

import os, re, json, math, pickle, time, shutil
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional

# Azure DI (Form Recognizer v4)
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

# Embeddings / RAG bits
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
from sklearn.preprocessing import minmax_scale

from config import (
    check_credentials,
    get_paths,
    WRITE_CHUNK_FILES,
    MAX_RETRIES_LIBRARIAN,
    MIN_HEADINGS_PER_CHUNK,
    MAX_HEADINGS_PER_CHUNK,
    DENSE_EMBEDDING_MODEL,
    RERANKER_MODEL,
    AZURE_DI_KEY,
    AZURE_DI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)

# =========================
# STEP 1: PDF -> RAW MD
# =========================
def extract_text_from_pdf(pdf_path: str, output_md_path: str) -> bool:
    print(f"Starting OCR for: {pdf_path}")
    if not AZURE_DI_KEY or not AZURE_DI_ENDPOINT:
        print("❌ Error: Azure Document Intelligence credentials not found in .env file.")
        return False
    try:
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT, credential=AzureKeyCredential(AZURE_DI_KEY)
        )
        with open(pdf_path, "rb") as pdf_file:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_file.read(),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
            result = poller.result()
        markdown_content = getattr(result, "content", "").strip()
        if not markdown_content:
            raise RuntimeError("❌ OCR returned empty content")
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        print(f"✅ Raw Markdown content saved to {output_md_path}")
        return True
    except FileNotFoundError:
        print(f"❌ Error: Input PDF not found at {pdf_path}")
    except Exception as e:
        print(f"❌ An error occurred during Azure DI processing: {e}")
    return False

# =========================
# STEP 2: refine MD headings
# =========================
def extract_marker(line: str, tag: str) -> Optional[str]:
    try:
        if not line:
            return None
        m = re.match(rf'^\s*<!--\s*{tag}="([^"]+)"\s*-->\s*$', line.strip())
        return m.group(1).strip() if m else None
    except Exception:
        return None

def classify_heading_level(text: str) -> int:
    try:
        if text.isupper():
            return 1
        elif all(w and w[0].isupper() for w in text.split()):
            return 2
        elif text.islower():
            return 4
        else:
            return 3
    except Exception:
        return 3

def refine_markdown_headings(input_md_path: str, output_md_path: str) -> bool:
    try:
        with open(input_md_path, "r", encoding="utf-8") as md_file:
            original_lines = md_file.readlines()
        if not original_lines:
            with open(output_md_path, "w", encoding="utf-8") as md_file:
                md_file.write("")
            return True

        header_positions, header_contents = [], []
        for idx, line in enumerate(original_lines):
            content = extract_marker(line, "PageHeader")
            if content:
                header_positions.append(idx); header_contents.append(content)

        headers_are_continuous = False
        if header_positions:
            expected_range = set(range(header_positions[0], header_positions[-1] + 1))
            actual_range = set(header_positions)
            if expected_range == actual_range and len(set(header_contents)) == 1:
                headers_are_continuous = True

        lines: List[str] = []
        for idx, line in enumerate(original_lines):
            stripped = line.strip()
            if not stripped or stripped == "<!-- PageBreak -->":
                continue
            header_content = extract_marker(stripped, "PageHeader")
            if header_content:
                if headers_are_continuous:
                    lines.append(header_content + "\n")
                else:
                    level = classify_heading_level(header_content)
                    lines.append(f"{'#' * level} {header_content}\n")
                continue
            if extract_marker(stripped, "PageFooter"):
                continue
            lines.append(line)

        processed, locked_first_line = [], None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if locked_first_line is None:
                locked_first_line = stripped
                level = classify_heading_level(locked_first_line)
                if not stripped.startswith("#"):
                    processed.append(f"{'#' * level} {locked_first_line}\n")
                else:
                    processed.append(line)
                continue
            processed.append(line)
            if stripped.lower() == "</table>":
                if i + 1 < len(lines) and lines[i + 1].strip():
                    processed.append("\n")

        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.writelines(processed)
        print(f"✅ Refined Markdown saved to {output_md_path}")
        return True
    except FileNotFoundError:
        print(f"❌ Error: Input Markdown file not found at {input_md_path}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# =========================
# STEP 3: chunk planning
# =========================
def call_llm_api(prompt: str, content: Optional[List[str]] = None) -> str:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        print("❌ Error: Azure OpenAI credentials not found in .env file.")
        return "{}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    payload = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n".join(content or [])},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
        "temperature": 0,
        "top_p": 0.95,
    }
    print("--- 🧠 Calling Azure OpenAI API (agent: range proposer)...")
    try:
        resp = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        j = resp.json()
        if j.get("choices"):
            msg = j["choices"][0].get("message", {}).get("content")
            if msg:
                print("✅ LLM response received.")
                return msg
        print(f"❌ Unexpected LLM response format: {j}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Azure OpenAI error: {e}")
        if getattr(e, "response", None) is not None:
            try: print(f"    Error Response: {e.response.text}")
            except Exception: pass
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
    return "{}"

def extract_table_of_contents(md_path: str) -> List[str]:
    print(f"Extracting Table of Contents from {md_path}...")
    toc: List[str] = []
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("#"):
                    toc.append(s)
        print(f"✅ Found {len(toc)} headings.")
        return toc
    except FileNotFoundError:
        print(f"❌ Error: Refined markdown file not found at {md_path}")
        return []
    except Exception as e:
        print(f"❌ TOC extraction failed: {e}")
        return []

def _number_headings(full_toc: List[str]) -> List[Dict]:
    try:
        return [{"n": i + 1, "text": h} for i, h in enumerate(full_toc)]
    except Exception:
        return [{"n": i + 1, "text": h} for i, h in enumerate(full_toc or [])]

def _indices_to_text(numbered: List[Dict], indices: List[int]) -> List[str]:
    try:
        idx_map = {h["n"]: h["text"] for h in numbered}
        return [idx_map[i] for i in indices if i in idx_map]
    except Exception:
        return []

def _strip_hashes(text: str) -> str:
    try:
        return re.sub(r'^\s*#+\s*', '', text).strip()
    except Exception:
        return text or ""

def _heading_level(h: str) -> int:
    try:
        m = re.match(r'^\s*(#+)', h)
        return len(m.group(1)) if m else 6
    except Exception:
        return 6

def _validate_ranges_cover_and_order(ranges: List[Tuple[int,int]], N: int) -> Tuple[bool,str]:
    try:
        if not ranges:
            return False, "No ranges."
        ranges = sorted(ranges, key=lambda r: (r[0], r[1]))
        cur = 1
        for (a,b) in ranges:
            if not (1 <= a <= b <= N):
                return False, f"Invalid range [{a},{b}] vs 1..{N}"
            if a != cur:
                return False, f"Gap/overlap before [{a},{b}] (expected start {cur})."
            cur = b + 1
        if cur != N + 1:
            return False, f"Did not end at {N}. Stopped at {cur-1}."
        return True, ""
    except Exception as e:
        return False, f"Validation failed: {e}"

def _enforce_size_bounds(ranges, min_sz, max_sz, levels):
    try:
        out = []
        for (a,b) in ranges:
            length = b - a + 1
            if length <= max_sz:
                out.append((a,b)); continue
            start = a
            while start <= b:
                target_end = min(start + max_sz - 1, b)
                best_end = target_end
                for j in range(target_end, start, -1):
                    if levels[j-1] <= min(4, levels[start-1]+1):
                        best_end = j; break
                if best_end < start: best_end = target_end
                out.append((start, best_end)); start = best_end + 1
        merged = []
        i = 0
        while i < len(out):
            a,b = out[i]; length = b - a + 1
            if length >= min_sz or i == len(out)-1:
                merged.append((a,b)); i += 1; continue
            na, nb = out[i+1]
            if (nb - a + 1) <= max_sz:
                merged.append((a, nb)); i += 2
            else:
                merged.append((a,b)); i += 1
        return merged
    except Exception:
        return ranges

def _proposer_prompt(numbered, target_min, target_max) -> str:
    listing = "\n".join(f"{h['n']}. {h['text']}" for h in numbered)
    return f"""
You are a careful librarian. Group the Markdown HEADING LINES into small, coherent, CONTIGUOUS ranges.
Use ONLY the heading text (no body). A "range" is inclusive: start_n..end_n and must be contiguous (no gaps inside).

Hard rules:
- Cover EVERY heading 1..N exactly once. No gaps, overlaps, or duplicates.
- Ranges MUST be contiguous and ordered: next range starts at previous end + 1.
- Prefer small precise ranges of about {target_min}–{target_max} headings per range.
- Single-heading ranges only when unavoidable (e.g., cover page, isolated heading).

JSON ONLY:
{{
  "chunks": [
    {{ "title": "<short generic topic>", "start_n": <int>, "end_n": <int> }}
  ]
}}

HEADINGS:
{listing}
""".strip()

def _repair_prompt(numbered, chunks, feedback, target_min, target_max) -> str:
    listing = "\n".join(f"{h['n']}. {h['text']}" for h in numbered)
    proposed = json.dumps({"chunks": chunks}, ensure_ascii=False)
    return f"""
Your previous contiguous ranges failed validation.

FEEDBACK:
{feedback}

Fix the ranges so that:
- Every heading 1..N appears exactly once with contiguous, non-overlapping ranges.
- Prefer small precise ranges of about {target_min}–{target_max} headings.
- Keep order strictly increasing.

Return JSON ONLY:
{{ "chunks": [{{ "title": "<short>", "start_n": <int>, "end_n": <int> }}] }}

CURRENT:
{proposed}

HEADINGS:
{listing}
""".strip()

def _parse_llm_ranges(s: str) -> List[Dict]:
    try:
        data = json.loads(s)
        ch = data.get("chunks") or []
        out = []
        for x in ch:
            a = int(x.get("start_n")); b = int(x.get("end_n"))
            title = (x.get("title") or "").strip() or "Topic"
            if a <= b:
                out.append({"title": title, "start_n": a, "end_n": b})
        return out
    except Exception:
        return []

def _choose_anchor_level(levels: List[int], min_target: int, max_target: int) -> int:
    try:
        N = len(levels)
        candidates = sorted(set(levels)) or [1]
        best_level = min(candidates); best_score = float("inf")
        for L in candidates:
            anchors = [i+1 for i,l in enumerate(levels) if l <= L]
            if anchors and anchors[0] != 1: anchors = [1] + anchors
            if not anchors: anchors = [1]
            segs = []
            for i,a in enumerate(anchors):
                b = (anchors[i+1]-1) if i+1 < len(anchors) else N
                segs.append((a,b))
            avg = sum(b-a+1 for a,b in segs)/len(segs)
            mid = (min_target+max_target)/2
            score = abs(avg - mid) + 0.5*len(segs)
            if score < best_score:
                best_score, best_level = score, L
        return best_level
    except Exception:
        return 2

def _fallback_ranges(levels: List[int], min_target: int, max_target: int) -> List[Tuple[int,int]]:
    try:
        N = len(levels)
        L = _choose_anchor_level(levels, min_target, max_target)
        anchors = [i+1 for i,l in enumerate(levels) if l <= L]
        if not anchors or anchors[0] != 1: anchors = [1] + anchors
        ranges = []
        for i,a in enumerate(anchors):
            b = (anchors[i+1]-1) if i+1 < len(anchors) else N
            ranges.append((a,b))
        ranges = _enforce_size_bounds(ranges, min_target, max_target, levels)
        ok,msg = _validate_ranges_cover_and_order(ranges, N)
        if not ok:
            ranges = []
            start = 1
            while start <= N:
                end = min(start + max_target - 1, N)
                ranges.append((start, end)); start = end + 1
        return ranges
    except Exception:
        N = len(levels)
        return [(1, N if N > 0 else 1)]

def build_chunk_plan(md_path: str, output_json_path: str,
                     max_retries_librarian: int = 4,
                     max_headings_per_chunk: int = 8,
                     min_headings_per_chunk: int = 2):
    try:
        full_toc = extract_table_of_contents(md_path)
        if not full_toc: return None
        numbered = _number_headings(full_toc)
        idx_to_text = {h["n"]: h["text"] for h in numbered}
        N = len(numbered)
        levels = [_heading_level(h["text"]) for h in numbered]

        llm_json = call_llm_api(_proposer_prompt(numbered, min_headings_per_chunk, max_headings_per_chunk))
        chunks = _parse_llm_ranges(llm_json)

        def _to_ranges(chs: List[Dict]): return [(c["start_n"], c["end_n"]) for c in chs]
        ok = False; msg = ""
        if chunks: ok, msg = _validate_ranges_cover_and_order(_to_ranges(chunks), N)
        if not ok:
            print(f"⚠️ Initial LLM ranges invalid: {msg}")
            for attempt in range(1, max_retries_librarian + 1):
                print(f"🔁 LLM repair attempt {attempt}/{max_retries_librarian}...")
                llm_json2 = call_llm_api(_repair_prompt(numbered, chunks, msg, min_headings_per_chunk, max_headings_per_chunk))
                chunks2 = _parse_llm_ranges(llm_json2)
                if chunks2:
                    ok2, msg2 = _validate_ranges_cover_and_order(_to_ranges(chunks2), N)
                    chunks, ok, msg = chunks2, ok2, msg2
                    if ok: print("✅ LLM repair succeeded."); break

        def _too_many_singletons(chs: List[Dict]) -> bool:
            if not chs: return True
            singles = sum(1 for c in chs if (c["end_n"] - c["start_n"] + 1) == 1)
            avg = sum((c["end_n"]-c["start_n"]+1) for c in chs)/len(chs)
            return singles > 0.25*len(chs) or avg < 1.8

        if not ok or not chunks or _too_many_singletons(chunks):
            print("🔧 Using structural fallback (heading-level based contiguous ranges).")
            ranges = _fallback_ranges(levels, min_headings_per_chunk, max_headings_per_chunk)
            chunks = []
            for (a,b) in ranges:
                title = _strip_hashes(idx_to_text.get(a, ""))
                chunks.append({"title": title if title else "Section", "start_n": a, "end_n": b})
            ok, msg = _validate_ranges_cover_and_order([(c["start_n"], c["end_n"]) for c in chunks], N)
            if not ok: print(f"❌ Fallback failed: {msg}")

        ranges = [(c["start_n"], c["end_n"]) for c in chunks]
        ranges = _enforce_size_bounds(ranges, min_headings_per_chunk, max_headings_per_chunk, levels)
        chunks = [{"title": _strip_hashes(idx_to_text.get(a, "")) or "Section", "start_n": a, "end_n": b} for (a,b) in ranges]

        ok_final, msg_final = _validate_ranges_cover_and_order([(c["start_n"], c["end_n"]) for c in chunks], N)
        validation_status = "passed" if ok_final else "failed"
        if not ok_final: print(f"❌ Final validation failed: {msg_final}")

        document_title = _strip_hashes(numbered[0]["text"]) if numbered else "Document"
        final_json = {
            "file_type": "chunk_plan",
            "document_title": document_title,
            "total_headings": N,
            "chunks": [],
            "validation": {"status": validation_status}
        }
        for k, ch in enumerate(chunks, start=1):
            a, b = ch["start_n"], ch["end_n"]
            hnums = list(range(a, b+1))
            final_json["chunks"].append({
                "chunk_index": k,
                "topic_title": ch["title"],
                "start_n": a,
                "end_n": b,
                "start_heading": idx_to_text.get(a, ""),
                "end_heading": idx_to_text.get(b, ""),
                "headings_n": hnums,
                "headings": _indices_to_text(numbered, hnums),
            })
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        print(f"\nSaving chunk plan to {output_json_path}...")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        print("✅ Chunk plan build complete.")
        return final_json
    except Exception as e:
        print(f"❌ Chunk plan build failed: {e}")
        return None

# =========================
# STEP 4: slice chunks
# =========================
def _index_headings_with_line_numbers(md_path: str) -> List[Tuple[int,int]]:
    try:
        indices: List[Tuple[int,int]] = []
        n = 0
        with open(md_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.lstrip().startswith("#"):
                    n += 1; indices.append((n, i))
        return indices
    except Exception as e:
        print(f"❌ Failed to index headings: {e}")
        return []

def slice_markdown_into_chunks(refined_md_path: str, chunk_plan: Dict, out_dir: str) -> Dict:
    try:
        if not chunk_plan or "chunks" not in chunk_plan:
            print("❌ No valid chunk plan provided to slicing step.")
            return {}
        os.makedirs(out_dir, exist_ok=True)
        with open(refined_md_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        heading_positions = _index_headings_with_line_numbers(refined_md_path)
        pos_map = {n: ln for (n, ln) in heading_positions}
        total_headings = chunk_plan.get("total_headings", 0)
        manifest = {"document_title": chunk_plan.get("document_title", "Document"),
                    "total_headings": total_headings, "chunks": []}
        for ch in chunk_plan["chunks"]:
            k = ch["chunk_index"]; a, b = int(ch["start_n"]), int(ch["end_n"])
            start_ln = pos_map.get(a); next_ln = pos_map.get(b + 1, len(all_lines))
            if start_ln is None:
                print(f"⚠️ Missing start line for heading {a}; skipping chunk {k}."); continue
            slice_lines = all_lines[start_ln:next_ln]
            fname = f"chunk_{k:03d}_{a}-{b}.md"; fpath = os.path.join(out_dir, fname)
            with open(fpath, "w", encoding="utf-8") as out:
                out.writelines(slice_lines)
            manifest["chunks"].append({"chunk_index": k, "file": fpath,
                                       "start_n": a, "end_n": b,
                                       "topic_title": ch.get("topic_title")})
        man_path = os.path.join(out_dir, "manifest.json")
        with open(man_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)
        print(f"✅ Wrote {len(manifest['chunks'])} chunk files to {out_dir}")
        return manifest
    except FileNotFoundError:
        print(f"❌ Error: Refined markdown file not found at '{refined_md_path}'.")
        return {}
    except Exception as e:
        print(f"❌ Slicing failed: {e}")
        return {}

# =========================
# STEP 5: embeddings + FAISS
# =========================
def create_vector_store(chunks_dir: str, vector_store_path: str, metadata_path: str, embedding_model: str):
    try:
        if not os.path.exists(chunks_dir):
            print(f"❌ Error: Chunks directory not found at '{chunks_dir}'."); return False
        chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.md')]
        if not chunk_files:
            print(f"❌ Error: No chunk files found in '{chunks_dir}'."); return False
        all_chunks = []
        for file_name in sorted(chunk_files):
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                all_chunks.append({"file_name": file_name, "content": f.read()})
        print(f"✅ Loaded {len(all_chunks)} chunks from disk.")
        print(f"Loading embedding model: {embedding_model}...")
        model = SentenceTransformer(embedding_model)
        print("Generating embeddings... (This may take a moment)")
        chunk_contents = [chunk['content'] for chunk in all_chunks]
        embeddings = model.encode(chunk_contents, show_progress_bar=True)
        if isinstance(embeddings, list): embeddings = np.array(embeddings)
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings).astype('float32'))
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        faiss.write_index(index, vector_store_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4)
        print(f"✅ FAISS index saved to '{vector_store_path}'")
        print(f"✅ Metadata saved to '{metadata_path}'")
        return True
    except Exception as e:
        print(f"❌ Vector store creation failed: {e}")
        return False

# =========================
# STEP 6: Hybrid retrieval + answer
# =========================
def _tokenize(txt: str) -> List[str]:
    try:
        return re.findall(r"[A-Za-z0-9\-_.]+", (txt or "").lower())
    except Exception:
        return []

def _mmr(doc_embs: np.ndarray, query_emb: np.ndarray, lambda_mult=0.5, top_k=8):
    try:
        if doc_embs.ndim != 2:
            doc_embs = np.asarray(doc_embs).reshape(len(doc_embs), -1)
        if query_emb.ndim != 1:
            query_emb = np.asarray(query_emb).ravel()
        N = doc_embs.shape[0]
        if N == 0: return []
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        sim_q = doc_embs @ q
        selected, candidates = [], list(range(N))
        while candidates and len(selected) < min(top_k, N):
            if not selected:
                best_local = int(np.argmax(sim_q[candidates]))
                selected.append(candidates.pop(best_local)); continue
            selected_embs = doc_embs[selected]
            sim_to_selected = doc_embs[candidates] @ selected_embs.T
            redundancy = sim_to_selected.max(axis=1)
            mmr_score = sim_q[candidates] - lambda_mult * redundancy
            best_local = int(np.argmax(mmr_score))
            selected.append(candidates.pop(best_local))
        return selected
    except Exception:
        return []

class HybridIndex:
    def __init__(self, metadata_path: str, faiss_path: str, bm25_pickle_path: str,
                 dense_model_name: str = DENSE_EMBEDDING_MODEL, reranker_model: str = RERANKER_MODEL):
        self.metadata_path = metadata_path
        self.faiss_path = faiss_path
        self.bm25_pickle = bm25_pickle_path
        self.dense_model_name = dense_model_name
        self.reranker_model = reranker_model
        self.dense_model = None
        self.faiss_index = None
        self.meta = []
        self.corpus_tokens = None
        self.bm25 = None
        self.reranker = None

    def load(self):
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.meta: List[Dict] = json.load(f)
            self.faiss_index = faiss.read_index(self.faiss_path)
            self.dense_model = SentenceTransformer(self.dense_model_name)
            if os.path.exists(self.bm25_pickle):
                with open(self.bm25_pickle, "rb") as f:
                    self.corpus_tokens, self.bm25 = pickle.load(f)
            else:
                self._build_bm25()
            try:
                self.reranker = CrossEncoder(self.reranker_model)
            except Exception:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✅ Hybrid index loaded."); return True
        except FileNotFoundError as e:
            print(f"❌ HybridIndex load error (file missing): {e}")
            return False
        except Exception as e:
            print(f"❌ HybridIndex load failed: {e}")
            return False

    def _build_bm25(self):
        try:
            texts = []
            for x in self.meta:
                content = x["content"]
                heading = re.findall(r"^#+\s+.*", content, flags=re.M)
                heading_text = " ".join([re.sub(r"^#+\s*", "", h) for h in heading]) if heading else ""
                boosted = (heading_text + " ") * 3 + content
                texts.append(boosted)
            self.corpus_tokens = [_tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            with open(self.bm25_pickle, "wb") as f:
                pickle.dump((self.corpus_tokens, self.bm25), f)
        except Exception as e:
            print(f"❌ BM25 build failed: {e}")
            self.corpus_tokens, self.bm25 = [], BM25Okapi([[]])

    def _azure_chat(self, system_prompt: str, user_content: str) -> str:
        try:
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                return ""
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 400,
                "temperature": 0.2,
                "top_p": 0.95
            }
            headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
            r = requests.post(AZURE_OPENAI_ENDPOINT, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            j = r.json()
            return j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception:
            return ""

    def expand_queries(self, q: str, n=3) -> Dict[str, List[str]]:
        try:
            sys = ("You are a retrieval assistant. Given a question, produce:\n"
                   "1) 3 short reformulations focusing on synonyms/abbreviations.\n"
                   "2) A 2-3 sentence hypothetical answer (HyDE) capturing likely facts.\n"
                   "Return JSON with keys: rewrites:[...], hyde:\"...\".")
            out = self._azure_chat(sys, q)
            rewrites, hyde = [], ""
            try:
                j = json.loads(out)
                rewrites = [q] + [r for r in j.get("rewrites", []) if isinstance(r, str)]
                hyde = j.get("hyde", "")
            except Exception:
                rewrites = [q]; hyde = ""
            dedup = []
            for r in rewrites:
                r = (r or "").strip()
                if r and r not in dedup: dedup.append(r)
            return {"rewrites": dedup[:n], "hyde": hyde}
        except Exception:
            return {"rewrites": [q], "hyde": ""}

    def _dense_search(self, queries: List[str], top_k=50):
        try:
            q_embs = self.dense_model.encode(queries, normalize_embeddings=True)
            q_vec = np.mean(q_embs, axis=0, keepdims=True)
            D, I = self.faiss_index.search(q_vec.astype("float32"), top_k)
            return I[0], (1.0/(1.0 + D[0]))
        except Exception:
            return np.array([], dtype=int), np.array([])

    def _bm25_search(self, queries: List[str], top_k=200):
        try:
            tokens = []
            for q in queries: tokens += _tokenize(q)
            scores = self.bm25.get_scores(tokens)
            idx = np.argsort(scores)[::-1][:top_k]
            return idx, scores[idx]
        except Exception:
            return np.array([], dtype=int), np.array([])

    def retrieve(self, user_query: str, k: int = 8) -> List[Dict]:
        try:
            expand = self.expand_queries(user_query, n=3)
            rewrites = expand["rewrites"]
            hyde = expand["hyde"]
            if hyde: rewrites = rewrites + [hyde]

            bm_idx, bm_sc = self._bm25_search(rewrites, top_k=200)
            de_idx, de_sc = self._dense_search(rewrites, top_k=100)

            fused_scores = {}
            if bm_sc.size > 0:
                bm_norm = minmax_scale(bm_sc) if len(bm_sc) > 1 else bm_sc
                for i, s in zip(bm_idx, bm_norm):
                    fused_scores[int(i)] = fused_scores.get(int(i), 0.0) + 0.6 * float(s)
            if de_sc.size > 0:
                de_norm = minmax_scale(de_sc) if len(de_sc) > 1 else de_sc
                for i, s in zip(de_idx, de_norm):
                    fused_scores[int(i)] = fused_scores.get(int(i), 0.0) + 0.8 * float(s)

            cand = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:50]
            if not cand: return []
            cand_idx = np.array([c[0] for c in cand], dtype=int)

            doc_embs = self.dense_model.encode([self.meta[i]["content"] for i in cand_idx], normalize_embeddings=True)
            q_emb    = self.dense_model.encode([user_query], normalize_embeddings=True)[0]
            mmr_order = _mmr(doc_embs, q_emb, lambda_mult=0.5, top_k=min(20, len(cand_idx)))
            if len(mmr_order) == 0: return []
            mmr_idx = cand_idx[mmr_order]

            pairs = [(user_query, self.meta[i]["content"][:2000]) for i in mmr_idx]
            ce_scores = self.reranker.predict(pairs)
            reranked = [(int(i), float(s)) for i, s in zip(mmr_idx, ce_scores)]
            reranked.sort(key=lambda x: x[1], reverse=True)
            final_idx = [i for i,_ in reranked[:k]]

            results = []
            for i in final_idx:
                item = dict(self.meta[i])
                item["chunk_id"] = int(i)
                item["score"] = fused_scores.get(int(i), 0.0)
                item["preview"] = self._snippet(item["content"], user_query)
                item["breadcrumbs"] = self._breadcrumbs(item["content"])
                results.append(item)
            return results
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []

    def _breadcrumbs(self, content: str) -> str:
        try:
            heads = re.findall(r"^#+\s+.*", content, flags=re.M)[:3]
            trail = [re.sub(r"^#+\s*", "", h).strip() for h in heads]
            return " > ".join([h for h in trail if h])
        except Exception:
            return ""

    def _snippet(self, content: str, query: str, window=240) -> str:
        try:
            q = re.escape(query.split()[0]) if query.split() else ""
            m = re.search(q, content, flags=re.I)
            if not m:
                return (content[:window] + "...") if len(content) > window else content
            start = max(0, m.start()-window//2); end = min(len(content), start+window)
            return ("..." if start>0 else "") + content[start:end] + ("..." if end<len(content) else "")
        except Exception:
            return content[:window] + ("..." if len(content) > window else "")

def answer_with_citations(index: HybridIndex, question: str, k_ctx=6) -> Dict:
    try:
        hits = index.retrieve(question, k=k_ctx)
        if not hits or (len(hits) and hits[0]["score"] < 0.1):
            return {"answer": "I can’t find a reliable answer in this manual.",
                    "citations": [], "confidence": 0.1}
        ctx_blocks = []
        for h in hits:
            ctx_blocks.append(f"[Chunk {h['chunk_id']}] {h['breadcrumbs']}\n{h['content'][:1800]}")
        context = "\n\n---\n\n".join(ctx_blocks)
        sys = (
            "You are a technical manual assistant. Answer ONLY with facts found in the provided context.\n"
            "Rules: cite chunk IDs inline like [Chunk 12]; if unknown, say you don’t know.\n"
            "Return JSON with keys: answer, citations (list of chunk_ids), confidence (0..1)."
        )
        user = f"Question:\n{question}\n\nContext:\n{context}"
        payload = {"messages":[{"role":"system","content":sys},{"role":"user","content":user}],
                   "temperature":0, "max_tokens":500, "response_format":{"type":"json_object"}}
        headers = {"Content-Type":"application/json","api-key":AZURE_OPENAI_API_KEY}
        r = requests.post(AZURE_OPENAI_ENDPOINT, json=payload, headers=headers, timeout=90)
        r.raise_for_status()
        try:
            j = r.json().get("choices", [{}])[0]["message"]["content"]
            out = json.loads(j)
        except Exception:
            out = {"answer":"(generation failed)","citations":[h['chunk_id'] for h in hits[:3]],"confidence":0.3}
        return out
    except requests.exceptions.RequestException as e:
        print(f"❌ Answer generation error: {e}")
        return {"answer":"(generation failed: API error)","citations":[],"confidence":0.2}
    except Exception as e:
        print(f"❌ Answering failed: {e}")
        return {"answer":"(generation failed)","citations":[],"confidence":0.2}

# =========================
# Helpers for Streamlit UI
# =========================
from time import time

def build_index_stepwise(pdf_base_name: str) -> Tuple[Optional[HybridIndex], Dict[str,str], Dict[str,object]]:
    """Run the full pipeline step-by-step (same logic, just exposed for UI)."""
    t_all0 = time()
    timings = {}
    stats = {}

    paths = get_paths(pdf_base_name)
    pdf_input = paths["pdf_input"]
    raw_md = paths["raw_md"]
    refined_md = paths["refined_md"]
    chunk_plan_json = paths["chunk_plan"]
    chunks_dir = paths["chunks_dir"]
    vector_store_path = paths["vector_store"]
    metadata_path = paths["metadata"]

    if not os.path.exists(pdf_input):
        print(f"❌ PDF not found at: {pdf_input}")
        return None, paths, {}

    t0 = time()
    if not extract_text_from_pdf(pdf_input, raw_md):
        return None, paths, {}
    timings["extract_text"] = time() - t0

    t0 = time()
    if not refine_markdown_headings(raw_md, refined_md):
        return None, paths, {}
    timings["refine_headings"] = time() - t0

    t0 = time()
    plan = build_chunk_plan(
        refined_md, chunk_plan_json,
        max_retries_librarian=MAX_RETRIES_LIBRARIAN,
        max_headings_per_chunk=MAX_HEADINGS_PER_CHUNK,
        min_headings_per_chunk=MIN_HEADINGS_PER_CHUNK,
    )
    timings["plan_chunks"] = time() - t0
    if not plan:
        return None, paths, {}

    t0 = time()
    manifest = slice_markdown_into_chunks(refined_md, plan, chunks_dir)
    timings["slice_chunks"] = time() - t0
    if not manifest or not manifest.get("chunks"):
        print("❌ No chunks produced; stopping.")
        return None, paths, {}

    t0 = time()
    ok = create_vector_store(
        chunks_dir=chunks_dir,
        vector_store_path=vector_store_path,
        metadata_path=metadata_path,
        embedding_model=DENSE_EMBEDDING_MODEL
    )
    timings["vector_store"] = time() - t0
    if not ok:
        return None, paths, {}

    t0 = time()
    idx = HybridIndex(
        metadata_path=metadata_path,
        faiss_path=vector_store_path,
        bm25_pickle_path=paths["bm25_pickle"],
        dense_model_name=DENSE_EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL
    )
    if not idx.load():
        return None, paths, {}
    timings["load_index"] = time() - t0

    # Stats for insights panel
    stats = {
        "document_title": plan.get("document_title", "Document"),
        "total_headings": plan.get("total_headings", 0),
        "chunk_count": len(manifest.get("chunks", [])),
        "validation": (plan.get("validation", {}) or {}).get("status", ""),
        "embedding_model": DENSE_EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "timings": timings
    }
    timings["total"] = time() - t_all0
    return idx, paths, stats


def delete_document_artifacts(pdf_base_name: str) -> bool:
    """Delete input PDF and all outputs for a base name."""
    try:
        paths = get_paths(pdf_base_name)
        targets = [
            paths["pdf_input"], paths["raw_md"], paths["refined_md"], paths["chunk_plan"],
            paths["vector_store"], paths["metadata"], paths["bm25_pickle"]
        ]
        for p in targets:
            try:
                if p and os.path.exists(p): os.remove(p)
            except Exception: pass
        try:
            if os.path.exists(paths["chunks_dir"]):
                shutil.rmtree(paths["chunks_dir"])
        except Exception: pass
        return True
    except Exception:
        return False
