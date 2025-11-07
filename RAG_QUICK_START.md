# RAG Quick Start (Public)

## What RAG Adds

Semantic similarity search over prior high-quality posts to give the LLM contextual examples (few-shot prompting).

Core files:
```
backend/rag_service.py          # Vector DB & retrieval
backend/ml_model_service_rag.py # Generation + post-process
backend/chroma_db/              # Persistent store (ignored by git)
```

LLM: Groq Llama-3.1-8b (free) preferred; OpenAI fallback.

API: Single POST `/api/optimize` returns array of `{style, post}` (hook, concise, rephrased).

---

## Setup (Once)
```powershell
pip install chromadb sentence-transformers groq
cd backend
C:/Users/adity/Downloads/VS_Code/python.exe rag_service.py  # build vector DB
```

---

## Enable High Quality

```powershell
$env:GROQ_API_KEY = "gsk_your-key-here"
cd backend
C:/Users/adity/Downloads/VS_Code/python.exe app.py
```
Log should mention Groq Llama-3.1-8b enabled.

---

## Quality Snapshot

Before (no RAG / templates): generic, weak hashtags.

---

After (RAG + Groq): specific examples, CTA, relevant emojis & hashtags.

---

## Improvements

Hooks, specificity, CTA, hashtags, length, tone all upgraded.

---

## Optional Next Steps

Immediate: set Groq key.

Future: expand dataset, engagement scoring, A/B tests.

---

## File Summary

See backend for RAG service and persisted `chroma_db/`.

---

## Test Without Key

Fallback templates (lower quality) will appear.

---

## Test With Key

Expect higher quality structured outputs for all three styles.

---

## Cost

Groq: free (beta) fast.

OpenAI: paid after free credits.

Recommendation: start with Groq.

---

## Flow

Draft → embed → retrieve similar → prompt LLM with examples → generate → post-process → response.

Why: in-context examples guide style + reduce hallucination; post-processing enforces quality rules.

---

## Troubleshooting

Backend fails: ensure Groq key set and port free.

Empty vector DB: run `rag_service.py`.

Low quality: verify Groq key and retrieval results (non-empty examples).

---

## Support

See detailed guides:
- `RAG_SETUP_GUIDE.md` (details)
- `QUICK_START.md` (commands)

RAG layer ready; set Groq key for best results.
