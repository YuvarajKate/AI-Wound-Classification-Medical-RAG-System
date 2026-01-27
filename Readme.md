# 🩺 AI Wound Classification & Medical RAG System

An end-to-end **Flask-based AI application** that:
- Classifies wound images using a **CNN (TensorFlow)**
- Provides **medical guidance** using **RAG (Retrieval-Augmented Generation)** with a local LLM

Designed for **exam, viva, and technical interviews**.

---

## 🔹 System Architecture (Diagrammatic)

User (Browser)
├── Upload Image / Image URL
└── Ask Medical Question
↓
Flask Application (app.py)
├── CNN Model (TensorFlow)
│ ↓
│ Wound Classification
│
└── RAG Pipeline (LangChain)
↓
Chroma Vector Database
↓
Ollama LLM (LlamaMedicine)
↓
Medical Guidance

---

## 🔹 Supported Wound Classes

Abrasions
Bruises
Burns
Cut
Ingrown_nails
Laceration
Stab_wound
Healthy

---

## 🔹 Technology Stack

| Layer | Technology |
|-----|-----------|
| Backend | Flask |
| ML Model | TensorFlow (CNN) |
| RAG Framework | LangChain |
| Vector DB | Chroma |
| LLM | Ollama (Elixpo/LlamaMedicine) |
| Embeddings | nomic-embed-text |
| Frontend | HTML (Jinja Templates) |

---

## 🔹 Project Structure

.
├── app.py # Main Flask app
├── wound_classifier_final.keras # Trained CNN model
├── class_names.json # Wound labels
├── uploads/ # Uploaded images
├── medical_knowledge_db/ # Medical PDFs
├── chroma_db/ # Vector store
├── templates/
│ └── index.html # UI
└── README.md

---

## 🔹 Working Explained

### Image Classification Flow

Input Image
↓
Resize (224 × 224)
↓
CNN Model
↓
Softmax Layer
↓
Predicted Wound Type

---

### Medical RAG Flow

User Question
↓
Text Embedding
↓
Chroma Similarity Search
↓
Relevant PDF Context
↓
Ollama LLM
↓
Context-Based Medical Answer

---

## 🔹 Why RAG Instead of Plain LLM?

- Prevents hallucinations
- Answers only from **trusted medical PDFs**
- Safer for healthcare-related use cases
- Strong architectural choice for interviews

---

## 🔹 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Start Ollama
ollama run Elixpo/LlamaMedicine
3. Run Application
python app.py
4. Open in Browser
http://127.0.0.1:5000
🔹 Key Interview Points
CNN handles visual understanding
RAG handles knowledge grounding
Chroma enables semantic search
Ollama allows local, private LLM inference
Clean separation of ML and NLP pipelines
🔹 Future Enhancements
Confidence score visualization
Multilingual medical responses
Mobile-first UI
Doctor-verified response layer
⚠️ Disclaimer
This project is for educational purposes only.
It is not a substitute for professional medical advice.

---

If you want next:
- 🔹 **One-page viva notes**
- 🔹 **System design explanation (2–3 min answer)**
- 🔹 **Interview Q&A from this project**

Just say the word.
