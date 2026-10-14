import os
import numpy as np
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies , CACHE_DIR
import re , json

class ChunkMeta (TypedDict):
    movie_idx:      int
    chunk_idx:      int
    total_chunks:   int

class SearchResult (TypedDict):
    score: float
    title: str
    description: str

class SemanticSearchResult (TypedDict):
    chunk_idx:  int
    movie_idx:  int
    score:      float

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None  #Movie embeddings list
        self.documents = None   #Movie Array
        self.document_map = {}  #Movie ID -> Movie 

    def generate_embedding(self,text: str) -> list:
        if text == "" or text is None:
            raise ValueError("Empty text to be embedded")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self , documents:list[dict]):
        self.documents = documents

        movies = []
        for doc in documents:
            self.document_map[doc['id']] = doc

            movies.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movies , show_progress_bar = True)
        np.save(CACHE_DIR + "/movie_embeddings.npy" , self.embeddings)

        return self.embeddings
        
    def load_or_create_embeddings(self , documents:list[dict]):
        
        if os.path.exists(CACHE_DIR + "/movie_embeddings.npy"):
            self.documents = documents

            for doc in documents:
                self.document_map[doc['id']] = doc
            
            stored_embeddings = np.load(CACHE_DIR + "/movie_embeddings.npy")
            if len(stored_embeddings) == len(documents):
                self.embeddings = stored_embeddings
                return self.embeddings
            

        return self.build_embeddings(documents)

    def search(self , query:str , limit:int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_vector = self.generate_embedding(query)

        results = []
        for i , doc in enumerate(self.documents , 0):
            similarity = cosine_similarity(query_vector , self.embeddings[i])
            results.append((similarity , doc))

        results = sorted(results , key= lambda item : item[0] , reverse=True)

        results = results[:limit]

        top_results = []
        for res in results:
            score = res[0]
            document = res[1]
            result = SearchResult(score=score , title=document['title'] , description= document['description'])
            
            top_results.append(result)
        
        return top_results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self , documents:list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        chunks = []
        chunk_meta = []

        for i , doc in enumerate(self.documents,0):
            text = doc['description']
            if text is None or text == "":
                continue
            movie_idx = i

            curr_chunks = semantic_chunk_text(text , 4 , 1)
            for idx , c in enumerate(curr_chunks,0):
                chunks.append(c)
                meta = ChunkMeta(
                        movie_idx=movie_idx,
                        chunk_idx=idx,
                        total_chunks=len(curr_chunks)
                    )
                chunk_meta.append(meta)

        self.chunk_embeddings = self.model.encode(chunks , show_progress_bar = True)
        self.chunk_metadata = chunk_meta

        np.save(CACHE_DIR + "/chunk_embeddings.npy" , self.chunk_embeddings)

        with open (CACHE_DIR + "/chunk_metadata.json" , "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents:list[dict]) -> np.ndarray:
        file1 = os.path.exists(CACHE_DIR + "/chunk_embeddings.npy")
        file2 = os.path.exists(CACHE_DIR + "/chunk_metadata.json")
        
        if file1 and file2:
            
            self.documents = documents
            for doc in documents:
                self.document_map[doc['id']] = doc
            
            self.chunk_embeddings = np.load(CACHE_DIR + "/chunk_embeddings.npy")

            with open(CACHE_DIR + "/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings
            

        return self.build_chunk_embeddings(documents)

    def search_chunks(self , query:str , limit:int = 10):
        query_vector = self.generate_embedding(query)

        results = []
        for i,ce in enumerate(self.chunk_embeddings,0):
            score = cosine_similarity(query_vector , ce)
            chunk_meta = self.chunk_metadata[i]
            res = SemanticSearchResult(
                chunk_idx = chunk_meta['chunk_idx'],
                movie_idx = chunk_meta['movie_idx'],
                score     = score 
            )
            results.append(res)

        best = {} #movie_idx -> bestChunkScore
        for res in results:
            movie_idx = res['movie_idx']
            score = res['score']
            if movie_idx not in best or score > best[movie_idx]['score']:
                best[movie_idx] = res
        
        best = sorted(best.items() , key = lambda item : item[1]['score'] , reverse = True)
        best = best[:limit]

        return_results = []
        for item in best:
            doc = self.documents[item[0]]
            return_results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(item[1]['score'], 3),
                }
            )
        
        return return_results



def verify_model() -> None:
    sc = SemanticSearch()

    print(f"Model loaded: {sc.model}")
    print(f"Max sequence length: {sc.model.max_seq_length}")

def embed_text(text:str) -> list:
    sc = SemanticSearch()
    embedding = sc.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sc = SemanticSearch()

    movies = load_movies()

    embeddings = sc.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query:str) -> list:
    sc = SemanticSearch()

    embedding = sc.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def fixed_chunk_text(text:str , chunk_size:int , overlap:int) -> list:
    text = text.split()
    chunks = []
    current_chunk = []

    for word in text:
        current_chunk.append(word)

        if len(current_chunk) == chunk_size:
            chunks.append(" ".join(current_chunk))

            current_chunk = current_chunk[-overlap:] if overlap > 0 else []

    if current_chunk and (not chunks or len(current_chunk) > overlap):
        chunks.append(" ".join(current_chunk))
    
    return chunks

def semantic_chunk_text(text:str , chunk_size:int , overlap:int):
    text = text.strip()
    if text is None or text == "":
        return []
    
    text_sentences = re.split(r"(?<=[.!?])\s+" , text)
    if len(text_sentences) == 1:
        ending_char = text_sentences[0][len(text_sentences[0])-1]
        if ending_char != '.' and ending_char != '!' and ending_char != '?':
            text_sentences = [text]

    chunks = []
    current_chunk = []

    for sentence in text_sentences:
        sentence = sentence.strip()
        if sentence == "":
            continue

        current_chunk.append(sentence)
        if len(current_chunk) == chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
    
    if current_chunk and (not chunks or len(current_chunk) > overlap):
        chunks.append(" ".join(current_chunk))

    return chunks
