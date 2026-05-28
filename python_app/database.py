import threading
import time
import sqlite3
import json
from typing import List, Dict, Any, Tuple
import core_vectordb

# Initialize SQLite database
DB_PATH = "vectordb.sqlite3"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meta TEXT,
    cat TEXT,
    embedding TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    text TEXT,
    embedding TEXT
)
''')
conn.commit()


class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store = {}
        self.bf = core_vectordb.BruteForce()
        self.kdt = core_vectordb.KDTree(dims)
        self.hnsw = core_vectordb.HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1
        
        # Reload existing vectors from SQLite
        self._reload_from_db()

    def _reload_from_db(self):
        cursor.execute("SELECT id, meta, cat, embedding FROM vectors")
        rows = cursor.fetchall()
        dist_fn = core_vectordb.get_dist_fn("cosine")
        max_id = 0
        for row in rows:
            vid, meta, cat, emb_str = row
            emb = json.loads(emb_str)
            v_item = core_vectordb.VectorItem(vid, meta, cat, emb)
            self.store[vid] = v_item
            self.bf.insert(v_item, dist_fn)
            self.kdt.insert(v_item, dist_fn)
            self.hnsw.insert(v_item, dist_fn)
            if vid > max_id:
                max_id = vid
        if max_id >= self.next_id:
            self.next_id = max_id + 1

    def insert(self, meta: str, cat: str, emb: List[float], metric: str = "cosine") -> int:
        with self.lock:
            emb_str = json.dumps(emb)
            cursor.execute("INSERT INTO vectors (meta, cat, embedding) VALUES (?, ?, ?)", (meta, cat, emb_str))
            conn.commit()
            vid = cursor.lastrowid
            
            if vid >= self.next_id:
                self.next_id = vid + 1

            v_item = core_vectordb.VectorItem(vid, meta, cat, emb)
            self.store[vid] = v_item
            
            dist_fn = core_vectordb.get_dist_fn(metric)
            self.bf.insert(v_item, dist_fn)
            self.kdt.insert(v_item, dist_fn)
            self.hnsw.insert(v_item, dist_fn)
            return vid

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            cursor.execute("DELETE FROM vectors WHERE id=?", (id,))
            conn.commit()
            
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            # KDTree rebuild
            items = list(self.store.values())
            self.kdt.rebuild(items)
            return True

    def search(self, q: List[float], k: int, metric: str = "cosine", algo: str = "hnsw") -> Dict[str, Any]:
        with self.lock:
            dist_fn = core_vectordb.get_dist_fn(metric)
            t0 = time.time()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn_ef(q, k, max(50, k), dist_fn)

            latency_us = int((time.time() - t0) * 1_000_000)

            hits = []
            for dist, vid in raw:
                if vid in self.store:
                    item = self.store[vid]
                    hits.append({
                        "id": vid,
                        "metadata": item.metadata,
                        "category": item.category,
                        "embedding": item.emb,
                        "distance": dist
                    })
            
            return {
                "hits": hits,
                "latency_us": latency_us,
                "algo": algo,
                "metric": metric
            }

    def benchmark(self, q: List[float], k: int, metric: str = "cosine") -> Dict[str, Any]:
        with self.lock:
            dist_fn = core_vectordb.get_dist_fn(metric)
            
            t0 = time.time()
            self.bf.knn(q, k, dist_fn)
            bf_us = int((time.time() - t0) * 1_000_000)
            
            t0 = time.time()
            self.kdt.knn(q, k, dist_fn)
            kd_us = int((time.time() - t0) * 1_000_000)
            
            t0 = time.time()
            self.hnsw.knn_ef(q, k, max(50, k), dist_fn)
            hnsw_us = int((time.time() - t0) * 1_000_000)
            
            return {
                "bf_us": bf_us,
                "kd_us": kd_us,
                "hnsw_us": hnsw_us,
                "n": len(self.store)
            }

    def all(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [{"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb} 
                    for v in self.store.values()]

    def hnsw_info(self) -> Dict[str, Any]:
        with self.lock:
            info = self.hnsw.get_info()
            return {
                "topLayer": info.topLayer,
                "nodeCount": info.nodeCount,
                "nodesPerLayer": info.nodesPerLayer,
                "edgesPerLayer": info.edgesPerLayer,
            }

    def size(self) -> int:
        with self.lock:
            return len(self.store)


class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = core_vectordb.HNSW(16, 200)
        self.bf = core_vectordb.BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0
        
        self._reload_from_db()

    def _reload_from_db(self):
        cursor.execute("SELECT id, title, text, embedding FROM documents")
        rows = cursor.fetchall()
        dist_fn = core_vectordb.get_dist_fn("cosine")
        max_id = 0
        for row in rows:
            vid, title, text, emb_str = row
            emb = json.loads(emb_str)
            if self.dims == 0:
                self.dims = len(emb)
                
            doc_item = {"id": vid, "title": title, "text": text, "emb": emb}
            self.store[vid] = doc_item
            
            v_item = core_vectordb.VectorItem(vid, title, "doc", emb)
            self.hnsw.insert(v_item, dist_fn)
            self.bf.insert(v_item, dist_fn)
            
            if vid > max_id:
                max_id = vid
                
        if max_id >= self.next_id:
            self.next_id = max_id + 1

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
                
            emb_str = json.dumps(emb)
            cursor.execute("INSERT INTO documents (title, text, embedding) VALUES (?, ?, ?)", (title, text, emb_str))
            conn.commit()
            vid = cursor.lastrowid
            
            if vid >= self.next_id:
                self.next_id = vid + 1
            
            doc_item = {"id": vid, "title": title, "text": text, "emb": emb}
            self.store[vid] = doc_item
            
            # Use 'doc' as category, title as metadata
            v_item = core_vectordb.VectorItem(vid, title, "doc", emb)
            dist_fn = core_vectordb.get_dist_fn("cosine")
            self.hnsw.insert(v_item, dist_fn)
            self.bf.insert(v_item, dist_fn)
            return vid

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Dict[str, Any]]:
        with self.lock:
            if not self.store:
                return []
            
            dist_fn = core_vectordb.get_dist_fn("cosine")
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn_ef(q, k, max(50, k), dist_fn)
                
            out = []
            for dist, vid in raw:
                if vid in self.store and dist <= max_dist:
                    out.append({"dist": dist, "doc": self.store[vid], "distance": dist})
            return out

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            cursor.execute("DELETE FROM documents WHERE id=?", (id,))
            conn.commit()
            
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)
