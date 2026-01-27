🩺 AI Wound Classification & Medical RAG System
This project is an end-to-end Flask-based AI application that integrates Computer Vision and Natural Language Processing. It classifies wound types using a Convolutional Neural Network (CNN) and provides contextual medical guidance using Retrieval-Augmented Generation (RAG) powered by a local LLM.

🏗️ System Architecture
The system follows a modular architecture separating visual perception from knowledge retrieval.

Workflow:

User Input: The user uploads a wound image or asks a medical question via a web interface.

Vision Pipeline: The Flask backend routes the image to a pre-trained TensorFlow CNN for classification.

Knowledge Pipeline: For queries, the LangChain framework performs a similarity search in a Chroma Vector DB to find relevant context from medical PDFs.

Generation: The Ollama (LlamaMedicine) LLM synthesizes the final answer based strictly on the retrieved context.

🛠️ Technology Stack
Layer	Technology	Role
Backend	Flask	Handles routing, image processing, and API logic.
Computer Vision	TensorFlow / Keras	A CNN model trained to classify 8 types of wounds.
RAG Framework	LangChain	Orchestrates the flow between the DB and the LLM.
Vector Database	Chroma DB	Stores high-dimensional embeddings of medical text.
Local LLM	Ollama (LlamaMedicine)	Generates safe, context-aware medical responses locally.
Embeddings	nomic-embed-text	Converts text into vectors for semantic search.
📂 Project Structure
Plaintext
.
├── app.py                      # Core Flask application & API routes
├── wound_classifier_final.keras # Trained CNN model (H5/Keras format)
├── class_names.json            # Mapping of indices to wound labels
├── medical_knowledge_db/       # Source Folder: Trusted Medical PDFs
├── chroma_db/                  # Persistent Vector Store
├── uploads/                    # Temporary storage for user-uploaded images
├── templates/                  # Frontend: Jinja2 HTML templates
└── requirements.txt            # Project dependencies
🔬 Core Components Explained
1. Image Classification (CNN)

The model takes an input image, resizes it to 224×224 pixels, and passes it through multiple convolutional and pooling layers to extract features. The final Softmax layer outputs probabilities for the following classes:

Abrasions, Bruises, Burns, Cut, Ingrown Nails, Laceration, Stab Wound, and Healthy Skin.

2. Medical RAG (Retrieval-Augmented Generation)

Standard LLMs can "hallucinate" (invent facts). To ensure safety:

Indexing: Medical PDFs are split into chunks and converted into vectors.

Retrieval: When a user asks a question, the system finds the most similar chunks in Chroma DB.

Grounding: The LLM is prompted: "Answer using ONLY the following context..." This ensures the advice is based on verified medical literature.

🚀 Setup & Execution
Install Dependencies:

Bash
pip install flask tensorflow langchain chromadb ollama
Initialize LLM:

Bash
ollama run Elixpo/LlamaMedicine
Launch App:

Bash
python app.py
Access the UI at http://127.0.0.1:5000.

🎓 Interview & Viva Key Points
Why Flask? It’s lightweight and ideal for deploying ML models without the overhead of larger frameworks.

Why CNN? CNNs are the gold standard for spatial feature extraction in images (detecting edges, textures, and wound patterns).

Why RAG? RAG provides traceability and reliability. Unlike a base LLM, we can point to the specific PDF page where the information originated.

Local Inference: By using Ollama, the medical data remains private and does not leave the local machine, addressing data privacy concerns in healthcare.

⚠️ Disclaimer: This project is for educational and demonstrative purposes. It is not a certified medical diagnostic tool.
