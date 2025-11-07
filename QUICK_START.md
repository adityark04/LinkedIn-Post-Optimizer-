# Quick Start - Run Project Now

## ✅ Current Status
Everything is already set up! Just start the servers.

## 🚀 Quick Start (2 Commands)

### Terminal 1 - Start Backend
```powershell
cd C:\Users\adity\Downloads\Aditya\College\Sem_7\DL\Replit\ai-linkedin-post-optimizer\backend
python app.py
```
**Wait for**: `* Running on http://127.0.0.1:5001`

### Terminal 2 - Start Frontend (New Terminal Window)
```powershell
cd C:\Users\adity\Downloads\Aditya\College\Sem_7\DL\Replit\ai-linkedin-post-optimizer
npm run dev
```
**Wait for**: `➜ Local: http://localhost:3000/`

### Open Browser
Go to: **http://localhost:3000**

---

## That's it! 🎉

Enter a LinkedIn post draft and click "Optimize Post" to see AI-generated suggestions.

---

## 📁 Clean Project Structure

The project has been cleaned to include only essential files:
- **15 root files** (HTML, TypeScript, config files, docs)
- **9 backend scripts** (app, models, training, data generation)
- **3 components** (Icons, Spinner, MarkdownRenderer)
- **1 service** (apiService for backend communication)
- **3 data files** (train, test, full datasets - 178 posts)
- **Trained models** (Transformer, CNN, T5)

All unused files removed for a cleaner codebase!

---

## If Models Need Training First

If backend shows model loading errors:

```powershell
cd C:\Users\adity\Downloads\Aditya\College\Sem_7\DL\Replit\ai-linkedin-post-optimizer\backend
python train_models.py
```

Wait 20-30 minutes for training to complete, then start servers as above.
