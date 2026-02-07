# 🩺 AI Wound Classification & Medical RAG System

An end-to-end **Flask-based AI application** that classifies wound images using a **EfficientNet (TensorFlow)** and provides grounded medical guidance using **RAG (Retrieval-Augmented Generation)**.

---
 
<code> 
Dataset Link - https://drive.google.com/drive/folders/1vgbEtBshfLRfwxig2wKiTb3mss5_PQt2?usp=sharing
</code>

## 🔹 System Architecture

The application utilizes a modular AI pipeline:
1.  **Vision Pipeline:** Processes image uploads through a **TensorFlow CNN** to identify 8 specific wound classes.
2.  **RAG Pipeline:** Uses **LangChain** and **ChromaDB** to retrieve context from trusted medical PDFs.
3.  **Inference Engine:** Generates safe, non-hallucinated responses via a local **Ollama LLM**.

---
![System Architecture](https://github.com/user-attachments/assets/e292d2ed-f4e8-40ea-87b6-c1d76375335a)


## 🔹 CNN Based Wound Classification
![CNN-Based Wound Classification Flow](https://github.com/user-attachments/assets/35c41e0a-b60f-433b-a613-ae8f1005cb10)

## 🔹 Medical RAG Pipeline Architecture
![Medical RAG Pipeline Architecture](https://github.com/user-attachments/assets/c78483d2-005c-4d87-a717-c796eeef43fd)

## 🔹 Overall Project Workflow
![Overall Project Workflow Diagram](https://github.com/user-attachments/assets/ad7d09d2-e864-4152-8f4a-e99e7363f424)

## 🔹 Technology Stack

| Layer | Technology | Role |
|:--- |:--- |:--- |
| **Backend** | Flask | API Orchestration & Routing |
| **ML Model** | TensorFlow | CNN-based Image Classification |
| **RAG Framework** | LangChain | Knowledge Retrieval Logic |
| **Vector DB** | Chroma | Semantic Search & Embeddings |
| **Containerization**| **Docker** | Environment Isolation & Portability |
| **LLM** | Ollama | Local Inference (LlamaMedicine) |

---





## 🔹 Supported Wound Classes
* Abrasions
* Bruises
* Burns
* Cut
* Ingrown_nails
* Laceration
* Stab_wound
* Healthy

---

## 🔹 Project Structure
```text
.
├── app.py                      # Main Flask Backend
├── Dockerfile                  # Container Configuration
├── wound_classifier_final.keras # Trained CNN Model
├── class_names.json            # Wound Label Mapping
├── medical_knowledge_db/       # Trusted Medical PDFs
├── chroma_db/                  # Persistent Vector Store
├── uploads/                    # User Uploaded Images
├── templates/                  # Frontend UI (Jinja2)
└── requirements.txt            # Dependencies
```


## 🔹 Deployment (Docker)
This project is containerized for production-ready consistency. It uses a Hybrid Architecture where the application logic is isolated in Docker while connecting to the host machine's LLM service.
</br>
To Build 
<code>
docker build -t <image_name> .
</code>
To Run
<code>
docker run -d --name woundapp \
  -p 5001:5000 \
  -e OLLAMA_BASE_URL="[http://host.docker.internal:11434](http://host.docker.internal:11434)" \
  -v "$(pwd)/medical_knowledge_db:/app/medical_knowledge_db" \
  -v "$(pwd)/chroma_db:/app/chroma_db" \
  -v "$(pwd)/uploads:/app/uploads" \
  wound-rag-app
</code>


## 🔹 Technical Implementation Details:

- Networking: host.docker.internal allows the containerized Flask app to communicate with the Ollama service running on the host OS.
- Volumes: Persistent storage is mounted for the medical_knowledge_db and chroma_db to ensure search indices remain intact during restarts.
- Environment Variables: The OLLAMA_BASE_URL allows for flexible LLM endpoint configuration without modifying code.


## 🔹 Key Technical Highlights for Interviews

- RAG over Plain LLM: Prevents medical hallucinations by forcing the model to answer based only on provided medical literature.
- Edge Privacy: By using Ollama, the system performs local inference, ensuring sensitive medical data never leaves the local environment.
- Model Optimization: The CNN handles spatial feature extraction (texture/edges), while the RAG pipeline handles semantic knowledge retrieval.
- DevOps Readiness: Full Dockerization ensures the "it works on my machine" problem is eliminated, providing a clean path to cloud deployment.

# 🔹 Authors
1. Yuvaraj Kate (yuvarajkate1740@gmail.com)
2. Dhamkirti Sisodia (dksisodia002@gmail.com)


