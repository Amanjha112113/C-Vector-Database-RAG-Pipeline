# Low-Level Design (LLD) Details

This document outlines the Low-Level Design of the VectorDB project, structured using SOLID principles with a C++ Core and Python API wrapper.

## Architecture Overview

1. **C++ Core (`core/`)**: Handles memory-intensive and CPU-bound algorithms (Distance Calculations, KDTree, HNSW, BruteForce). Compiled as a shared library (`core_vectordb`) using `pybind11`.
2. **Python API (`python_app/`)**: Handles network I/O, REST routing (FastAPI), JSON serialization, and coordination with Ollama for language models.

---

## C++ Core Layer

### `VectorItem` (`core/include/VectorItem.h`)
Data structure representing a single point in the vector space.
- **Attributes:**
  - `int id`: Unique identifier.
  - `string metadata`: Textual description or title.
  - `string category`: Grouping label (e.g., "cs", "math", "doc").
  - `vector<float> emb`: The actual vector embeddings.

### `Distance` (`core/include/Distance.h`)
Defines the distance metric functions.
- **Functions:**
  - `float euclidean(a, b)`: Calculates straight-line distance.
  - `float cosine(a, b)`: Calculates cosine distance (1 - cosine similarity).
  - `float manhattan(a, b)`: Calculates L1 norm (city block distance).
  - `DistFn getDistFn(string metric)`: Factory function returning the appropriate function pointer.

### `IVectorIndex` (`core/include/IVectorIndex.h`)
**Interface** demonstrating the Dependency Inversion Principle. All search algorithms must implement this contract.
- **Methods:**
  - `virtual void insert(VectorItem, DistFn) = 0;`
  - `virtual vector<pair<float, int>> knn(query, k, DistFn) = 0;`
  - `virtual void remove(int id) = 0;`

### `BruteForce` (`core/include/BruteForce.h`)
Implements exact Nearest Neighbor search by iterating over all stored vectors. Time complexity: O(N * d).
- **Attributes:**
  - `vector<VectorItem> items`: Linear list of all vectors.

### `KDTree` (`core/include/KDTree.h`)
Implements exact spatial partitioning search. Effective in low dimensions, degrades in high dimensions. Time complexity: O(log N).
- **Attributes:**
  - `KDNode* root`: The root of the binary space partitioning tree.
- **Methods:**
  - `void search(...)`: Recursively traverses the tree, pruning branches based on hypersphere bounds.
  - `void rebuild(...)`: Since deleting from a balanced KDTree is complex, this function reconstructs the tree when a node is removed.

### `HNSW` (Hierarchical Navigable Small World) (`core/include/HNSW.h`)
Implements production-grade Approximate Nearest Neighbor (ANN) search. High recall and low latency in extremely high dimensions (768+).
- **Attributes:**
  - `unordered_map<int, HNSWNode> G`: The multilayer graph mapping node IDs to their connections per layer.
  - `int M`, `M0`, `ef_build`: HNSW hyper-parameters controlling graph density and build quality.
  - `int topLayer`, `int entryPt`: The entry point into the graph at the sparsest top layer.
- **Methods:**
  - `searchLayer()`: Greedy beam search navigating connections within a specific layer.
  - `knn_ef()`: Overload of search allowing custom `ef` parameter for trade-off between speed and recall during search.

### `PybindWrapper` (`core/bindings/PybindWrapper.cpp`)
Uses `pybind11` to expose the C++ classes and data structures to the Python runtime without standard boilerplate. Binds `get_dist_fn`, `VectorItem`, `BruteForce`, `KDTree`, and `HNSW` as fully accessible Python objects in the `core_vectordb` module.

---

## Python Application Layer

### `llm_client.py`
Handles communication with the local Ollama daemon. 
- **Class `OllamaClient`:**
  - **Methods:**
    - `is_available()`: Pings Ollama.
    - `embed(text)`: Calls `nomic-embed-text` to generate a 768-D vector.
    - `generate(prompt)`: Calls `qwen2.5:3b` (or chosen model) for text generation.
- **Function `chunk_text()`:** Utility to split large documents into sliding windows of 250 words to fit context limits.

### `database.py`
Acts as the Manager/Facade pattern over the C++ Core objects. Adds thread-safety (`threading.Lock`) so the FastAPI server can handle concurrent requests safely.
- **Class `VectorDB`:**
  - Manages the 16-D demo vectors. Instantiates all three C++ indexes (`BruteForce`, `KDTree`, `HNSW`) simultaneously for benchmarking purposes.
- **Class `DocumentDB`:**
  - Manages the high-dimensional (768-D) vectors for the RAG pipeline. Primarily relies on `HNSW` for performance, keeping `BruteForce` only as a fallback for tiny datasets (<10 nodes).

### `main.py`
The FastAPI application.
- **Endpoints:**
  - `GET /search`: Takes a vector string, routes to the requested algorithm via `VectorDB.search()`.
  - `GET /benchmark`: Runs a query across all 3 algorithms and returns latency comparisons.
  - `POST /doc/insert`: Receives a document, chunks it, calls `OllamaClient.embed()`, and inserts into `DocumentDB`.
  - `POST /doc/ask`: The RAG endpoint. Embeds the user question, retrieves top-K chunks from `DocumentDB`, and builds a context prompt for `OllamaClient.generate()`.
  - `GET /status`: Health check for Ollama and DB sizes.
- **Static Files:** Mounts `index.html` from the `/static` directory to serve the frontend UI.
