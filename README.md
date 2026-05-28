# High-Performance C++ Vector Database & RAG Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.14%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)

A custom-built, ultra-low-latency Vector Database and Retrieval-Augmented Generation (RAG) system. Engineered from scratch, the core semantic search engine is written in C++ for maximum performance, bridged to a modern Python web backend using PyBind11, and integrated with local LLMs (Ollama) to ensure complete data privacy and offline capability.

## 🚀 Key Features

* **High-Performance C++ Engine:** Custom implementation of semantic similarity search algorithms entirely in C++ bypassing the overhead of Python.
* **Multiple Indexing Algorithms:** 
  * **HNSW (Hierarchical Navigable Small World):** O(log N) approximate nearest neighbor search for massive datasets.
  * **KD-Tree:** Efficient exact nearest neighbor search for lower-dimensional embeddings.
  * **Brute-Force (Flat):** Guaranteed exact nearest neighbor baseline with AVX/SIMD-ready distance calculations.
* **Local Generative AI:** Integrated with `Ollama` running `Qwen2.5` and `nomic-embed-text` locally. No API keys required, zero data sent to the cloud.
* **Persistent Storage:** SQLite-backed persistence layer. Your vectors, metadata, and documents survive server restarts without needing a heavy PostgreSQL/Redis deployment.
* **Modern Web Interface:** A decoupled HTML/CSS/JS frontend featuring a dark/light mode toggle, dynamic PCA visualization mapping, and real-time inference streaming.
* **Python Binding Architecture:** Utilizes `PyBind11` to compile the C++ codebase into a seamless Python module (`core_vectordb.so`), orchestrated by `FastAPI`.

---

## 🧠 Algorithmic Deep Dive

Why build a Vector Database from scratch? To understand the underlying mathematics of modern AI search.

### 1. HNSW (Hierarchical Navigable Small World)
HNSW is the industry standard for Approximate Nearest Neighbor (ANN) search. It builds a multi-layered graph where the top layers are sparse (fast traversal) and the bottom layers are dense (accurate local search). 
* **Complexity:** O(log N) search time.
* **Implementation Details:** Custom priority queues to maintain the closest `ef_construction` nearest neighbors during insertions.

### 2. KD-Tree
A spatial partitioning data structure that recursively splits space into half-spaces based on the median of the data along the axis with the highest variance.
* **Complexity:** O(log N) for low dimensions. 
* **Implementation Details:** Tree structure with backtracking pruning bounds.

### 3. Distance Metrics
The core math relies on optimized distance functions. Currently supports:
* **Cosine Similarity:** $1 - \frac{A \cdot B}{||A|| \cdot ||B||}$
* **Euclidean (L2) Distance:** $\sqrt{\sum (A_i - B_i)^2}$

---

## 🏗 System Architecture

The project follows a strict decoupled microservices-like architecture:

1. **`core/` (C++)**: The mathematical engine. Contains memory-managed `VectorItem` structs, distance metrics, and the search trees.
2. **`PyBindWrapper.cpp`**: Exposes the C++ classes (`HNSW`, `KDTree`, `BruteForce`) to Python securely.
3. **`database.py`**: The Data Access Layer. Handles the SQLite connection, JSON serialization, and orchestrates the Python-to-C++ calls. Rebuilds the C++ memory indices on startup from the persistent `.sqlite3` file.
4. **`llm_client.py`**: The AI API Wrapper. Makes asynchronous HTTP requests to the local Ollama instance on port `11434`.
5. **`main.py`**: The FastAPI controller. Exposes REST endpoints (`/doc/insert`, `/doc/ask`) and mounts the static frontend.
6. **`static/`**: The presentation layer. Responsive UI built with Vanilla JS and CSS.

---

## ⚙️ Installation & Setup

### Prerequisites
* **macOS / Linux** (Optimized for Apple Silicon / M-series chips)
* **Python 3.14+**
* **C++ Compiler** supporting C++17 (`clang` or `gcc`)
* **Ollama** installed globally (`brew install ollama`)

### 1. Start the Local LLM (Ollama)
Before running the backend, you must pull the required models and start the Ollama server.
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama serve
```

### 2. Set Up the Python Environment
Clone the repository and set up a virtual environment.
```bash
git clone https://github.com/Amanjha112113/C-Vector-Database-RAG-Pipeline.git
cd C-Vector-Database-RAG-Pipeline
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies & Compile C++
Install the required python packages (`FastAPI`, `Uvicorn`, `Pybind11`, `Requests`).
Then, compile the C++ core into a Python Shared Object (`.so`) file.
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### 4. Run the Backend
Start the FastAPI server.
```bash
cd python_app
uvicorn main:app --reload --port 8080
```
Visit `http://localhost:8080` in your web browser.

---

## 🛠 API Documentation

### `POST /doc/insert`
Embeds and inserts a new document into the database.
* **Payload:**
  ```json
  {
    "title": "Aman Jha",
    "text": "Aman Jha is an AI Engineering student."
  }
  ```
* **Response:**
  ```json
  {
    "status": "success",
    "doc_id": 1,
    "dims": 768
  }
  ```

### `POST /doc/ask`
Queries the database with a question and generates an LLM response based on the top retrieved context.
* **Payload:**
  ```json
  {
    "query": "Who is Aman Jha?",
    "k": 3
  }
  ```
* **Response:**
  ```json
  {
    "answer": "Based on the context, Aman Jha is an AI Engineering student...",
    "context_used": [
      {
        "title": "Aman Jha",
        "distance": 0.12
      }
    ],
    "latency_us": 1450,
    "algo": "hnsw"
  }
  ```

---

## 📂 Project Structure

```text
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py                 # PyBind11 Build Script
├── core/                    # C++ Core Engine
│   ├── include/
│   │   ├── BruteForce.h
│   │   ├── Distance.h
│   │   ├── HNSW.h
│   │   ├── IVectorIndex.h
│   │   ├── KDTree.h
│   │   └── VectorItem.h
│   ├── src/
│   │   ├── BruteForce.cpp
│   │   ├── Distance.cpp
│   │   ├── HNSW.cpp
│   │   └── KDTree.cpp
│   └── bindings/
│       └── PybindWrapper.cpp # Python Bridge
└── python_app/              # FastAPI Backend
    ├── main.py              # Endpoints
    ├── database.py          # SQLite & C++ Orchestration
    ├── llm_client.py        # Ollama API integration
    └── static/              # Frontend UI
        ├── index.html
        ├── style.css
        └── app.js
```

---

## 🤝 Contributing
Contributions are welcome! If you'd like to optimize the AVX-512 distance calculations, add DiskANN support, or improve the Vanilla JS UI, please submit a Pull Request.

## 📝 License
This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for details.
