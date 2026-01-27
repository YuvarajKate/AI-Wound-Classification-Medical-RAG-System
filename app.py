import os

# ✅ Must be at the TOP (before tensorflow import)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # 0=all logs, 3=errors only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # force CPU (avoids GPU spam/crash)

import json
import uuid
import logging
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# --- LangChain RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ✅ Silence tensorflow python logger too
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = Flask(__name__)

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = "./uploads"
DATA_PATH = "./medical_knowledge_db"  # Store your trusted PDFs here
CHROMA_PATH = "./chroma_db"

MODEL_PATH = "./wound_classifier_final.keras"
CLASS_NAMES_PATH = "./class_names.json"
IMG_SIZE = (224, 224)

# Ollama models
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Elixpo/LlamaMedicine")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ✅ Docker will use host ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Flask config
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ----------------------------
# Lazy-loaded global objects
# ----------------------------
vector_store = None
rag_chain = None
cnn_model = None

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


# ----------------------------
# Helpers
# ----------------------------
def download_image(url: str, save_dir: str) -> str:
    if not url.startswith("http"):
        raise ValueError("Invalid URL. Must start with http/https")

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    content_type = (r.headers.get("Content-Type") or "").lower()
    if "image" not in content_type:
        raise ValueError("URL does not return an image")

    ext = ".jpg"
    if "png" in content_type:
        ext = ".png"
    elif "webp" in content_type:
        ext = ".webp"

    filename = f"url_{uuid.uuid4().hex}{ext}"
    path = os.path.join(save_dir, filename)

    with open(path, "wb") as f:
        f.write(r.content)

    return path


def preprocess_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# ----------------------------
# RAG Setup
# ----------------------------
def setup_rag():
    print("✅ RAG: Loading PDFs...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if len(documents) == 0:
        raise RuntimeError(f"No PDFs found in {DATA_PATH}. Add PDFs first.")

    print(f"✅ RAG: PDF count = {len(documents)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    print("✅ RAG: Splitting documents...")
    chunks = splitter.split_documents(documents)
    print(f"✅ RAG: Total chunks = {len(chunks)}")

    print("✅ RAG: Creating embeddings + Chroma DB (first time can take time)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("✅ RAG: Vector DB created successfully")
    return vector_db


def load_or_build_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # If folder exists but is empty -> rebuild
    if (not os.path.isdir(CHROMA_PATH)) or (len(os.listdir(CHROMA_PATH)) == 0):
        return setup_rag()

    print("✅ RAG: Loading existing Chroma DB...")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def get_rag_chain():
    """
    ✅ Lazy initialization so Flask starts immediately.
    """
    global vector_store, rag_chain

    if rag_chain is not None:
        return rag_chain

    vector_store = load_or_build_vectorstore()

    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )

    print("✅ RAG: RetrievalQA chain ready")
    return rag_chain


def get_cnn_model():
    """
    ✅ Lazy initialization so Flask starts immediately.
    """
    global cnn_model

    if cnn_model is not None:
        return cnn_model

    print("✅ CNN: Loading model...")
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ CNN: Model loaded")
    return cnn_model


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "pdf_folder": DATA_PATH,
        "pdf_count": len([f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]),
        "chroma_folder": CHROMA_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_path = None

        # 1) File upload
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = secure_filename(file.filename)

            if filename == "":
                return jsonify({"error": "Invalid filename"}), 400

            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(img_path)

        # 2) URL input
        elif request.form.get("image_url"):
            url = request.form.get("image_url").strip()
            img_path = download_image(url, app.config["UPLOAD_FOLDER"])

        else:
            return jsonify({"error": "No image provided (file or image_url required)."}), 400

        # ✅ CNN Classification
        model = get_cnn_model()
        img_array = preprocess_image(img_path)
        preds = model.predict(img_array, verbose=0)[0]

        wound_type = CLASS_NAMES[int(np.argmax(preds))]
        confidence = float(np.max(preds))

        # ✅ RAG Execution (lazy load)
        rag = get_rag_chain()
        query = f"""
You are a medical first-aid assistant.
Give step-by-step first aid for: {wound_type}.
Make it short, actionable, safe, and clear.
If emergency signs exist, tell user to seek urgent care.
"""
        response = rag.invoke(query)
        first_aid = response.get("result", "No result returned.")

        return jsonify({
            "wound_type": wound_type,
            "confidence": round(confidence, 4),
            "first_aid": first_aid
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("✅ Flask is starting now...")
    app.run(debug=True, host="0.0.0.0", port=5000)
