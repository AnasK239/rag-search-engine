import os , pickle
from collections import Counter
from .search_utils import tokenize_text, load_movies , PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT + "/cache"

class InvertedIndex:
    def __init__(self) -> None:
        self.index = {} # token -> set of document ids
        self.docmap = {} # doc ID -> full document object
        self.term_frequencies = {} # doc Id -> Counter object

    def __add_document(self , doc_id, text):
        tokens = tokenize_text(text)

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
        
    def build(self):
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

    def load(self) -> None:

        with open(CACHE_DIR+"/index.pkl" , "rb") as f:
            self.index = pickle.load(f)

        with open(CACHE_DIR+"/docmap.pkl" , "rb") as f:
            self.docmap = pickle.load(f)

        with open(CACHE_DIR+"/term_frequencies.pkl" , "rb") as f:
            self.term_frequencies = pickle.load(f)