import os
import numpy as np
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies , CACHE_DIR


class SearchResult (TypedDict):
    score: float
    title: str
    description: str

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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