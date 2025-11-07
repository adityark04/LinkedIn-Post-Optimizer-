# Note: This long-form guide has been archived. For the up-to-date public instructions, see README.md and QUICK_START.md. The app now uses RAG + Groq; model training steps below are legacy.
# AI LinkedIn Post Optimizer - Complete Setup & Run Guide

## 📋 Prerequisites
- Python 3.11
- Node.js and npm
- Git (optional)

## 🚀 Complete Flow to Run the Project

### Step 1: Install Python Dependencies
```powershell
cd backend
pip install -r requirements.txt
```

**What this does**: Installs all Python packages (PyTorch, Flask, Transformers, etc.)

---

### Step 2: Install Frontend Dependencies
```powershell
cd ..
npm install
```

**What this does**: Installs React, Vite, and other frontend dependencies

---

### Step 3: Prepare Dataset (Already Done - Skip if you have 178 posts)
```powershell
cd backend

# Option A: Use existing 178-post dataset (RECOMMENDED - Already prepared)
# Files already exist:
# - data/train_data.json (142 posts)
# - data/test_data.json (36 posts)
# - data/full_dataset.json (178 posts)

# Option B: Generate fresh dataset (if needed)
python scrape_linkedin_bulk.py
# Choose option 2 to generate 100+ posts
python merge_scraped_data.py
```

**What this does**: Creates training dataset with 178 diverse LinkedIn posts

---

### Step 4: Train Models (First Time Setup)
```powershell
# Make sure you're in backend directory
cd backend

# Train all three models (takes 20-30 minutes)
python train_models.py
```

**What this does**: 
- Builds vocabulary (1335 tokens)
- Trains Transformer model for hook generation
- Trains CNN model for concise summaries  
- Fine-tunes T5 model for rephrasing
- Saves best models in `saved_models/` directory

**Expected Output**:
- Transformer: Val loss ~3.27 (46% improvement)
- CNN: Val loss ~6.4
- T5: Val loss ~3.8
- Models saved to `saved_models/`

---

### Step 5: Start Backend Server
```powershell
# In backend directory
cd backend
python app.py
```

**What this does**: 
- Loads trained models
- Starts Flask API server on http://127.0.0.1:5001
- Ready to receive optimization requests

**You should see**:
```
Loading trained models...
✓ Vocabulary loaded (size: 1335)
✓ Transformer model loaded
✓ CNN model loaded
✓ T5 model loaded
 * Running on http://127.0.0.1:5001
```

**Keep this terminal running!**

---

### Step 6: Start Frontend Server (New Terminal)
```powershell
# Open NEW terminal in project root
cd C:\Users\adity\Downloads\Aditya\College\Sem_7\DL\Replit\ai-linkedin-post-optimizer

npm run dev
```

**What this does**:
- Starts Vite dev server on http://localhost:3000
- Serves React frontend

**You should see**:
```
VITE v6.4.1  ready in 340 ms
➜  Local:   http://localhost:3000/
```

**Keep this terminal running!**

---

### Step 7: Use the Application
Open your browser and go to: **http://localhost:3000**

**How to use**:
1. Enter a LinkedIn post draft (e.g., "Excited to share that our team launched a new product")
2. Click "Optimize Post"
3. Get three AI-generated versions:
   - **Hook**: Engaging version with emoji
   - **Concise**: Short, punchy summary
   - **Rephrased**: Full optimized post with hashtags

---

## 🔧 Quick Commands Reference

### Daily Usage (After Initial Setup)

**Terminal 1 - Backend**:
```powershell
cd backend
python app.py
```

**Terminal 2 - Frontend**:
```powershell
npm run dev
```

**Then open**: http://localhost:3000

---

## 🛠️ Troubleshooting

### If Backend Fails to Load Models:
```powershell
cd backend

# Retrain models
python train_models.py
```

### If Vocabulary Size Mismatch Error:
```powershell
cd backend

# Quick CNN retrain
python retrain_cnn_quick.py
```

### If Port 5001 Already in Use:
```powershell
# Kill existing Python process
Get-Process python | Stop-Process -Force

# Restart backend
cd backend
python app.py
```

### If Port 3000 Already in Use:
```powershell
# Kill existing Node process  
Get-Process node | Stop-Process -Force

# Restart frontend
npm run dev
```

---

## 📊 Current Project Status

### Dataset
- **Total**: 178 LinkedIn posts
- **Training**: 142 posts (80%)
- **Testing**: 36 posts (20%)
- **Vocabulary**: 1335 tokens

### Models Status
- ✅ **Transformer** (Hook): Trained, Val loss 3.27
- ✅ **T5** (Rephrase): Trained, Val loss 3.86
- ✅ **CNN** (Concise): Trained, Val loss 6.40

### Servers
- **Backend**: http://127.0.0.1:5001
- **Frontend**: http://localhost:3000

---

## 📁 Project Structure

```
ai-linkedin-post-optimizer/
├── backend/
│   ├── app.py                          # Flask API server
│   ├── ml_model_service.py             # Model inference service
│   ├── train_models.py                 # Main training script
│   ├── scrape_linkedin_bulk.py         # Data generation
│   ├── merge_scraped_data.py           # Dataset merging
│   ├── retrain_cnn_quick.py            # Quick CNN retrain
│   ├── requirements.txt                # Python dependencies
│   ├── models/
│   │   ├── transformer_model.py        # Transformer architecture
│   │   ├── cnn_model.py                # CNN architecture
│   ├── data/
│   │   ├── train_data.json             # Training data (142)
│   │   ├── test_data.json              # Test data (36)
│   │   └── full_dataset.json           # All data (178)
│   └── saved_models/
│       ├── transformer_hook_best.pth   # Trained Transformer
│       ├── cnn_concise_best.pth        # Trained CNN
│       ├── t5_rephrase/                # Fine-tuned T5
│       └── vocab.pkl                   # Vocabulary
├── src/
│   ├── App.tsx                         # Main React component
│   ├── index.tsx                       # React entry point
│   └── services/
│       └── apiService.ts               # Backend API calls
├── package.json                        # Frontend dependencies
├── index.html                          # HTML template
└── vite.config.ts                      # Vite configuration
```

---

## 🎯 Testing the Models

### Test Generation Quality
```powershell
cd backend
python test_generation.py
```

This will test all models with sample drafts and show output quality.

---

## 📈 Retraining Models (When Adding New Data)

```powershell
cd backend

# 1. Generate/scrape new posts
python scrape_linkedin_bulk.py

# 2. Merge with existing data
python merge_scraped_data.py

# 3. Retrain all models
python train_models.py

# 4. Restart backend
python app.py
```

---

## ✅ Complete Start-to-Finish Commands

**First Time Setup** (Run once):
```powershell
# Install dependencies
cd backend
pip install -r requirements.txt
cd ..
npm install

# Train models (if not already trained)
cd backend
python train_models.py
```

**Every Time You Use the App**:
```powershell
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend (new terminal)
npm run dev

# Open browser to http://localhost:3000
```

---

## 🔍 Verify Everything Works

Run this checklist:

1. ✅ Backend dependencies installed: `cd backend; pip list | findstr torch`
2. ✅ Frontend dependencies installed: `npm list react`
3. ✅ Dataset exists: `ls backend\data\train_data.json`
4. ✅ Models trained: `ls backend\saved_models\*.pth`
5. ✅ Backend running: Visit http://127.0.0.1:5001 (should see "Not Found")
6. ✅ Frontend running: Visit http://localhost:3000 (should see UI)
7. ✅ Test optimization: Enter draft → click Optimize → see results

---

**That's it! You're ready to optimize LinkedIn posts with AI! 🚀**
