import os , pickle , math
from collections import Counter
from .search_utils import DEFAULT_SEARCH_LIMIT, tokenize_text, load_movies , PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT + "/cache"

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {} # token -> set of document ids
        self.docmap = {} # doc ID -> full document object
        self.term_frequencies = {} # doc Id -> Counter object
        self.doc_lengths = {}
        self.avg_doc_len = 0.0
        self.index_path = CACHE_DIR + "/index.pkl"

    def __get_avg_doc_length(self) -> float:
        total_docs = len(self.docmap)
        if total_docs == 0 :
            return 0.0
            
        total = 0
        for value in self.doc_lengths.values():
            total += value
            
        return float(total) / total_docs
        
    def __add_document(self , doc_id, text):
        tokens = tokenize_text(text)

        self.doc_lengths[doc_id] = len(tokens)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
            
        for tok in tokens:
            if tok not in self.index:
                self.index[tok] = set()
                
            self.term_frequencies[doc_id][tok] +=  1  
            self.index[tok].add(doc_id)


    def get_documents(self , term) -> list:
        return sorted(self.index.get(term , set()))

    def get_tf(self, doc_id , term):
        if term not in self.term_frequencies[doc_id]:
            return 0

        return self.term_frequencies[doc_id][term]
        
    def get_bm25_idf(self , term:str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))

        bm_25idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm_25idf

    def get_bm25_tf(self , doc_id , term , k1 = BM25_K1 , b = BM25_B) -> float:
        basic_tf = self.get_tf(doc_id , term)

        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.avg_doc_len
        
        doc_length_norm = 1 - b + b * (doc_length / avg_doc_length)
        
        bm25_tf = (basic_tf * (k1+1)) / (basic_tf + k1 * doc_length_norm)
        return bm25_tf

    def bm25(self , doc_id , term) -> float:
        return self.get_bm25_tf(doc_id , term) * self.get_bm25_idf(term)

    def bm25_search(self , query:str , limit = DEFAULT_SEARCH_LIMIT) -> list:
        query_tokens = tokenize_text(query)

        scores = {}
        for doc_id  in self.docmap.keys():
            total_bm25 = 0
            for tok in query_tokens:
                total_bm25 += self.bm25(doc_id , tok)

            if total_bm25 > 0:
                scores[doc_id] = total_bm25

        ranked = sorted(
            scores.items(),
            key= lambda item : item[1],
            reverse= True
        )

        return ranked[:limit]
        
    def build(self):
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
        self.doc_lengths.clear()
        
        movies = load_movies()

        for mov in movies:
            self.__add_document(mov['id'] , f"{mov['title']} {mov['description']}")
            self.docmap[mov['id']] = mov

    def save(self) -> None:
        os.makedirs(CACHE_DIR , exist_ok=True)

        with open(CACHE_DIR+"/index.pkl" , "wb") as f:
            pickle.dump(self.index , f)

        with open(CACHE_DIR+"/docmap.pkl" , "wb") as f:
            pickle.dump(self.docmap , f)

        with open(CACHE_DIR+"/term_frequencies.pkl" , "wb") as f:
            pickle.dump(self.term_frequencies , f)

        with open(CACHE_DIR+"/doc_lengths.pkl" , "wb") as f:
            pickle.dump(self.doc_lengths , f)

    def load(self) -> None:

        with open(CACHE_DIR+"/index.pkl" , "rb") as f:
            self.index = pickle.load(f)

        with open(CACHE_DIR+"/docmap.pkl" , "rb") as f:
            self.docmap = pickle.load(f)

        with open(CACHE_DIR+"/term_frequencies.pkl" , "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(CACHE_DIR+"/doc_lengths.pkl" , "rb") as f:
            self.doc_lengths = pickle.load(f)

        self.avg_doc_len = self.__get_avg_doc_length()