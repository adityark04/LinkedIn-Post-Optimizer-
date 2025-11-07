<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI LinkedIn Post Optimizer

A full-stack web application that uses **three custom-trained deep learning models** to optimize LinkedIn posts. This project demonstrates real-world applications of CNNs, Transformers, and Transfer Learning.

## 🎯 Project Overview

Transform your draft LinkedIn posts into three professionally optimized versions:

1. **Transformer-Generated** - Adds engaging hooks and attention-grabbing openings
2. **CNN-Enhanced** - Creates concise, punchy versions that get to the point
3. **T5-Rephrased** - Generates SEO-optimized, professional versions

### Deep Learning Models Used

- **Custom Transformer** (Encoder-Decoder with Multi-Head Attention)
- **CNN for Text** (1D Convolutions with Multiple Filter Sizes)
- **Fine-tuned T5** (Hugging Face Pre-trained Model)

**No external APIs or GPT services** - All models are trained and run locally!

---

## 🏗️ Architecture

```
Frontend (React + TypeScript + Vite)
    ↓
Flask REST API (Python)
    ↓
ML Model Service
    ├── Transformer Model (Hook Generation)
    ├── CNN Model (Concise Generation)
    └── T5 Model (Rephrasing)
```

---

## 📋 Prerequisites

- **Node.js** (v16 or higher)
- **Python** 3.8+ 
- **pip** (Python package manager)

---

## 🚀 Quick Start

### Step 1: Install Frontend Dependencies

```powershell
npm install
```

### Step 2: Install Backend Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

This installs PyTorch, Transformers, Flask, and other ML libraries.

### Step 3: Train the Models

**IMPORTANT:** You must train the models before running the application.

```powershell
# Generate dataset
python prepare_dataset.py

# Train all three models (takes ~20-30 minutes on CPU)
python train_models.py
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions.

### Step 4: Run the Application

**Terminal 1 - Start Backend:**
```powershell
cd backend
python app.py
```

**Terminal 2 - Start Frontend:**
```powershell
npm run dev
```

**Open your browser:** http://localhost:5173

---

## 📖 Detailed Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete guide to training models
- **Model architectures** - See `backend/models/` directory
- **API documentation** - See `backend/app.py`

---

## 🧠 Deep Learning Concepts Demonstrated

### 1. Transformer Model (`transformer_model.py`)
- ✅ Multi-head self-attention
- ✅ Positional encoding
- ✅ Encoder-decoder architecture
- ✅ Masked attention (prevents looking ahead)
- ✅ Beam search decoding

### 2. CNN Model (`cnn_model.py`)
- ✅ 1D Convolutions for text
- ✅ Multiple kernel sizes (3, 4, 5)
- ✅ Max pooling
- ✅ Feature extraction
- ✅ Sequence-to-sequence generation

### 3. T5 Model (Hugging Face)
- ✅ Transfer learning
- ✅ Fine-tuning pre-trained models
- ✅ Text-to-text transformation
- ✅ Conditional generation

---

## 📁 Project Structure

```
ai-linkedin-post-optimizer/
├── frontend/
│   ├── App.tsx              # Main React component
│   ├── services/
│   │   └── apiService.ts    # API calls to backend
│   └── components/          # UI components
├── backend/
│   ├── app.py              # Flask API server
│   ├── ml_model_service.py # Model loading and inference
│   ├── prepare_dataset.py  # Dataset generation
│   ├── train_models.py     # Training script
│   ├── models/
│   │   ├── transformer_model.py  # Custom Transformer
│   │   ├── cnn_model.py          # CNN for text
│   │   └── __init__.py
│   ├── data/               # Training data (generated)
│   ├── saved_models/       # Trained model weights
│   └── requirements.txt    # Python dependencies
├── TRAINING_GUIDE.md       # Detailed training guide
└── README.md              # This file
```

---

## 🎓 Educational Value

This project is perfect for learning:

1. **End-to-end ML pipeline** - From data preparation to deployment
2. **Multiple DL architectures** - CNNs, Transformers, Transfer Learning
3. **Full-stack integration** - Connecting ML models to web applications
4. **PyTorch fundamentals** - Model building, training, and inference
5. **Production considerations** - Model serving, error handling

---

## 🔧 Training Details

### Dataset
- Synthetic LinkedIn posts generated in `prepare_dataset.py`
- Each sample has 4 versions: draft, hook, concise, rephrased
- 80/20 train/test split

### Training Time (CPU)
- Transformer: ~10-15 minutes (30 epochs)
- CNN: ~8-10 minutes (30 epochs)
- T5: ~5-10 minutes (5 epochs)
- **Total: ~20-30 minutes**

### Model Sizes
- Vocabulary: ~200-500 words
- Transformer: ~2-3 MB
- CNN: ~1-2 MB
- T5: ~242 MB (pre-trained model)

---

## 🎨 Features

- ✅ Real-time post optimization
- ✅ Three different AI styles
- ✅ Copy-to-clipboard functionality
- ✅ Dark theme UI
- ✅ Responsive design
- ✅ Error handling
- ✅ Loading states

---

## 🛠️ Customization

### Add More Training Data

Edit `backend/prepare_dataset.py` to add more examples:

```python
drafts = [
    "Your new LinkedIn draft...",
    # Add more
]
```

Then retrain:
```powershell
python prepare_dataset.py
python train_models.py
```

### Adjust Model Architecture

Modify hyperparameters in `train_models.py`:

```python
model = TransformerHookGenerator(
    d_model=256,      # Increase for larger model
    nhead=8,          # More attention heads
    num_encoder_layers=3,  # Deeper network
)
```

---

## 🐛 Troubleshooting

### "Models Not Trained" Error
**Solution:** Run `python train_models.py` in the backend directory.

### NLTK Download Errors
**Solution:** 
```powershell
python -c "import nltk; nltk.download('punkt')"
```

### PyTorch Installation Issues
**Solution:** Use CPU-only version:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues During Training
**Solution:** Reduce batch size in `train_models.py` from 4 to 2 or 1.

---

## 📊 Model Performance

After training, models will:
- Generate contextually relevant hooks
- Create meaningful summaries
- Rephrase in professional LinkedIn style

**Note:** Performance improves with more training data!

---

## 🚀 Future Enhancements

- [ ] Add more training data (real LinkedIn posts)
- [ ] Implement user feedback loop
- [ ] Add model evaluation metrics
- [ ] GPU acceleration support
- [ ] Model comparison dashboard
- [ ] Export to LinkedIn directly

---

## 📝 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- PyTorch for deep learning framework
- Hugging Face for Transformers library
- React team for frontend framework

---

## 📧 Support

For issues or questions:
1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. Review error messages carefully
3. Ensure all dependencies are installed

---

**Happy Learning! 🎓**
