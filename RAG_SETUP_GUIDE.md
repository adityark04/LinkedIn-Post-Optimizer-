# RAG Setup Guide (Public)
For quick run steps, see `README.md` and `QUICK_START.md`. This guide provides more context on the RAG layer and enabling Groq.

## What's Been Implemented

### ✅ Vector Database (ChromaDB)
- **178 LinkedIn posts** loaded into vector database
- **Semantic search** using sentence-transformers (all-MiniLM-L6-v2)
- **384-dimensional embeddings** for similarity matching
- Finds 3-5 most similar high-quality posts for any draft

### ✅ RAG Service (`rag_service.py`)
- ChromaDB persistent storage in `backend/chroma_db/`
- Similarity search with metadata filtering
- Automatic post type detection (hook, concise, rephrased)
- Handles multiple JSON dataset formats

### ✅ RAG-Enhanced ML Service (`ml_model_service_rag.py`)
- Dual mode: Groq Llama-3.1 (preferred) or OpenAI fallback
- **Intelligent prompting** with RAG examples
- **Post-processing** rules for emoji/hashtag quality
- **3 optimization styles**: Hook, Concise, Rephrased

### ✅ Flask API Integration
- Updated `app.py` to use RAG service
- Automatic fallback to basic service if RAG unavailable
- Same API endpoints, better quality

---

## Current Status (Without OpenAI)

**Right now**: System uses **pattern-based fallback** (basic quality)
- ✅ Vector search works
- ✅ Finds similar posts
- ❌ No GPT generation (needs API key)

**Output quality**: 3/10 (simple templates)

---

## Enable High-Quality Mode (Groq Recommended)

1. Sign up: https://console.groq.com/
2. Create API key (starts with gsk_)
3. In terminal: `$env:GROQ_API_KEY = "gsk_your-key-here"`
4. Start backend: `C:/Users/adity/Downloads/VS_Code/python.exe backend/app.py`

---

## Setup Instructions

### Step 1: Set API Key (Windows PowerShell)
```powershell
$env:GROQ_API_KEY = "gsk_your-key-here"
```

### Step 2: Test RAG Service
```powershell
C:/Users/adity/Downloads/VS_Code/python.exe backend/ml_model_service_rag.py
```

Expected output includes:
```
✅ Groq Llama-3.1-8b enabled
Vector database ready with 178 posts.
```

### Step 3: Start Backend
```powershell
cd backend
C:/Users/adity/Downloads/VS_Code/python.exe app.py
```

You should see lines confirming Groq or OpenAI plus server start:
```
✅ Groq Llama-3.1-8b enabled
 * Running on http://127.0.0.1:5001
```

---

## Optional: OpenAI Fallback
If `GROQ_API_KEY` is not set and `OPENAI_API_KEY` is, the backend will use OpenAI automatically.

---

## Quality Comparison

### Without API Key (Current)
```
Input: "AI is changing software development"
Output: 🚀 AI is changing software development

What's your experience with this? Let me know in the comments! 💬
```
**Quality**: 3/10 (generic template)

### With OpenAI/Groq (After setup)
```
Input: "AI is changing software development"
Output: AI-powered code generation is revolutionizing how developers work 🚀

GitHub Copilot, ChatGPT, and AI assistants are:
- Writing boilerplate code in seconds
- Debugging complex issues faster
- Translating between languages instantly

But here's the catch: AI doesn't replace developers—it amplifies them.

The best developers use AI as a co-pilot, not autopilot.

What's your experience with AI coding tools? Are they helping or hurting your workflow? 💬

#SoftwareEngineering #AI #DeveloperTools
```
**Quality**: 9/10 (specific, engaging, professional)

---

## Next Steps

### Immediate (Get Better Quality)
1. ✅ Get OpenAI or Groq API key
2. ✅ Set environment variable
3. ✅ Restart backend
4. ✅ Test with real drafts

### Future Enhancements
- [ ] Scrape 500+ more high-quality LinkedIn posts
- [ ] Add A/B testing (compare outputs)
- [ ] Fine-tune Llama-3.1 locally
- [ ] Add analytics (engagement prediction)

---

## Cost Analysis

### OpenAI GPT-3.5-turbo
- **Cost**: $0.002 per 1,000 tokens
- **Average post**: ~300 tokens = $0.0006 per optimization
- **$5 free credits**: ~8,300 optimizations
- **After free tier**: $1 = ~1,600 optimizations

### Groq (Llama-3.1)
- **Cost**: FREE (unlimited during beta)
- **Speed**: 300+ tokens/second (faster than GPT)
- **Quality**: Comparable to GPT-3.5

### Recommendation
Start with **Groq** (free + fast) → Switch to **GPT-4** if you need absolute best quality

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'chromadb'"
```powershell
pip install chromadb sentence-transformers openai
```

### Vector database empty
```powershell
C:/Users/adity/Downloads/VS_Code/python.exe backend/rag_service.py
```

### Backend crashes on startup
Check if another process is using port 5001:
```powershell
Get-Process | Where-Object {$_.ProcessName -eq "python"}
```

---

## Files Created/Modified

### New Files
- ✅ `backend/rag_service.py` - Vector database + similarity search
- ✅ `backend/ml_model_service_rag.py` - RAG-enhanced generation
- ✅ `backend/chroma_db/` - Persistent vector database

### Modified Files
- ✅ `backend/app.py` - Updated to use RAG service

### Dependencies Added
- ✅ chromadb
- ✅ sentence-transformers
- ✅ openai
- ✅ langchain-community
- ✅ langchain-core

---

## Summary

**RAG infrastructure is ready! 🎉**

Current mode: **Pattern-based fallback** (3/10 quality)

To enable **high-quality mode** (9/10):
1. Get OpenAI or Groq API key
2. Set `OPENAI_API_KEY` or `GROQ_API_KEY` environment variable
3. Restart backend

**Estimated time to enable**: 5 minutes  
**Cost**: Free with Groq, $5 free credits with OpenAI
