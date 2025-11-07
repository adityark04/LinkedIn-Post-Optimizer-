# Quick Start

Get the app running locally in minutes.

## Prerequisites
- Node.js 18+
- Python 3.11 with pip

## 1) Install dependencies
```powershell
# From project root
npm install

# Backend deps
cd backend
pip install -r requirements.txt
```

## 2) Set your LLM key (Groq recommended)
```powershell
# New terminal session where you run the backend
$env:GROQ_API_KEY = "gsk_your-key-here"
```
Get a free key: see GET_GROQ_KEY.md

## 3) Start the servers
```powershell
# Terminal 1 – Backend
cd backend
C:/Users/adity/Downloads/VS_Code/python.exe app.py
```
```powershell
# Terminal 2 – Frontend (project root)
cd ..
npm run dev
```

Open http://localhost:3000 and paste your draft to get three optimized versions.

## Troubleshooting
- Backend won’t start: ensure `$env:GROQ_API_KEY` is set in the same terminal.
- Vector DB empty: run `C:/Users/adity/Downloads/VS_Code/python.exe backend/rag_service.py` once.
- More details: see README.md and RAG_SETUP_GUIDE.md.
