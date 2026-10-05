# PyRAG - Local RAG (Retrieval-Augmented Generation) CLI & Web App

PyRAG is a fully local, privacy-first Retrieval-Augmented Generation (RAG) system built with Python. It allows you to ingest local text documents into a vector database (PostgreSQL + pgvector or Weaviate) and chat with them using local LLMs via Ollama. 

It comes with both a powerful Command Line Interface (CLI) and a sleek Web UI.

## 🚀 Features
- **100% Local & Private:** No API keys required, no data leaves your machine. Powered by [Ollama](https://ollama.com/).
- **Hybrid Search:** Combines semantic (vector) and lexical (keyword) search using Reciprocal Rank Fusion (RRF) for highly accurate document retrieval.
- **Multimodal Support:** Ingest images and query them alongside text documents using vision models (e.g., LLaVA).
- **Auto-Ingestion:** Watch a folder for changes and automatically chunk & embed new files.
- **Multiple Vector Stores:** Supports PostgreSQL (`pgvector`) by default, with an optional Weaviate backend.
- **Interactive Web UI:** A modern web interface to upload documents, images, and chat with your data seamlessly.

---

## 📋 Prerequisites

Before installing PyRAG, ensure you have the following installed on your system:
1. **[Python 3.10+](https://www.python.org/downloads/)**
2. **[Docker & Docker Compose](https://docs.docker.com/get-docker/)** (for the database)
3. **[Ollama](https://ollama.com/)** (for running local LLMs)

### Setting up Ollama
PyRAG requires an embedding model and a chat model. Once Ollama is installed, open your terminal and run:

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Pull the chat model
ollama pull gemma3:latest

# Pull the vision model (for image ingestion)
ollama pull llava:7b

# Make sure the Ollama server is running in the background
ollama serve
```

---

## 🛠️ Installation & Setup

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd pyrag
```

**2. Set up the Python virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -e .
```

**4. Environment Variables:**
Create a `.env` file in the root directory (you can copy the example if one exists). Minimal `.env` setup:
```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=dummy
CHAT_MODEL=gemma3:latest
EMBED_MODEL=nomic-embed-text
VISION_MODEL=llava:7b
EMBED_DIM=768
VECTOR_STORE=postgres
```

**5. Start the Database:**
Spin up the PostgreSQL database (with `pgvector` extension) via Docker:
```bash
docker compose up -d
```
*(If you want to use Weaviate instead, update `VECTOR_STORE=weaviate` in `.env` and run `docker compose --profile weaviate up -d`)*

---

## 💻 Usage

PyRAG provides several commands through its CLI. You can type `pyrag --help` at any time to see available options.

### Ingesting Documents & Images
PyRAG processes text documents (`.txt`, `.md`, `.markdown`) as well as images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`).
```bash
# Ingest a single file
pyrag ingest documents/sample.txt

# Scan and ingest all files inside the documents/ folder
pyrag scan

# Start a background watcher that auto-ingests new or modified files
pyrag watch
```
*(Ingested files are automatically moved to the `documents/processed/` folder).*

### Querying & Chatting
```bash
# Ask a one-off question
pyrag ask "What is a vampire?"

# Start an interactive CLI chat session
pyrag chat
```

### Running the Web Interface
PyRAG includes a beautiful web interface. To start the server:
```bash
pyrag serve
```
Then, open your browser and navigate to: **http://127.0.0.1:8000**

---

## 🏗️ Project Structure
- `src/pyrag/cli.py`: The CLI application logic (built with Typer).
- `src/pyrag/web.py`: The FastAPI backend and SSE stream handler.
- `src/pyrag/static/`: Frontend HTML/JS/CSS for the Web UI.
- `src/pyrag/ingest.py`: Chunking and embedding pipeline + background directory watcher.
- `src/pyrag/query.py`: Hybrid search, Context retrieval and LLM prompt generation.
- `src/pyrag/stores/`: Vector Database adapters (Postgres & Weaviate).

---

## 🧹 Cleanup
To stop and remove the database container:
```bash
docker compose down
```
If you want to completely wipe the database and start fresh:
```bash
docker compose down -v
```
