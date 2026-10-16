import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

DEFAULT_WEIGHTED_ALPHA = 0.5
DEFAULT_LIMIT = 5
DEFAULT_K = 60   #RRF SCORING

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = DEFAULT_LIMIT) -> list[dict]:
        keyword_results = self._bm25_search(query , 500*limit)
        
        norm_bm25_scores = []
        for tuple in keyword_results:
            norm_bm25_scores.append(tuple[1])
        norm_bm25_scores = minmax_normalize(norm_bm25_scores)

        semantic_results = self.semantic_search.search_chunks(query , 500*limit)
        
        norm_semantic_scores = []
        for item in semantic_results:
            norm_semantic_scores.append(item['score'])
        norm_semantic_scores = minmax_normalize(norm_semantic_scores)

        #doc_id -> {"doc" : document , "keywordscore": 1 , "semanticscore":2 , "hybridscore" :3}
        hybrid_results = {} 

        for i,tuple in enumerate(keyword_results,0):
            doc_id = tuple[0]
            norm_score = norm_bm25_scores[i]
            res = {
                "doc" : self.idx.docmap[doc_id],
                "keyword_score" : norm_score,
                "semantic_score" : 0,
                "hybrid_score" : hybrid_score(norm_score , 0 , alpha)
            }
            hybrid_results[doc_id] = res

        for i , item in enumerate(semantic_results , 0):
            doc_id = item['id']
            norm_score = norm_semantic_scores[i]
            if doc_id not in hybrid_results:
                res = {
                    "doc" : self.idx.docmap[doc_id],
                    "keyword_score" : 0,
                    "semantic_score" : norm_score,
                    "hybrid_score" : hybrid_score(0 , norm_score , alpha)
                }
                hybrid_results[doc_id] = res
            else:
                keyword_score = hybrid_results[doc_id]['keyword_score']
                hybrid_results[doc_id]['semantic_score'] = norm_score
                hybrid_results[doc_id]['hybrid_score'] = hybrid_score(keyword_score , norm_score , alpha)
               
        
        sorted_results = sorted(hybrid_results.items() , key = lambda item:item[1]['hybrid_score'], reverse = True)
        
        return_res = []
        for id , map in sorted_results:
            res = {
                "id" : id,
                "title" : map['doc']['title'],
                "hybrid_score" : round(float(map['hybrid_score']) , 3),
                "bm25_score" : round(float(map['keyword_score']) , 3),
                "semantic_score" : round(float(map['semantic_score']) ,3),
                "description" : map['doc']['description'][:100]
            }
            return_res.append(res)


        return return_res[:limit]
    

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:

        keyword_results = self._bm25_search(query , 500*limit)
        bm25_rrf_ranks = []
        bm25_rrf_scores = []
        for i , tuple in enumerate(keyword_results,1):
            bm25_rrf_ranks.append(i)
            bm25_rrf_scores.append(rrf_score(i , k))

        semantic_results = self.semantic_search.search_chunks(query , 500*limit)
        semantic_rrf_ranks = [] 
        semantic_rrf_scores = []       
        for i , res in enumerate(semantic_results,1):
            semantic_rrf_ranks.append(i)
            semantic_rrf_scores.append(rrf_score(i , k))

        #doc_id -> {"doc" : document , "rrf_score": 1 , "semantic_rank":2 , "bm25_rank" :3}
        hybrid_fusion = {}

        for i,tuple in enumerate(keyword_results,0):
            doc_id = tuple[0]
            bm25_rank = bm25_rrf_ranks[i]
            bm25_rrf = bm25_rrf_scores[i]

            res = {
                "doc" : self.idx.docmap[doc_id],
                "bm25_rank" : bm25_rank,
                "semantic_rank" : None,
                "rrf_score" : bm25_rrf 
            }
            hybrid_fusion[doc_id] = res

        for i , item in enumerate(semantic_results , 0):
            doc_id = item['id']
            semantic_rank = semantic_rrf_ranks[i]
            semantic_rrf = semantic_rrf_scores[i]

            if doc_id not in hybrid_fusion:
                res = {
                    "doc" : self.idx.docmap[doc_id],
                    "bm25_rank" : None,
                    "semantic_rank" : semantic_rank,
                    "rrf_score" : semantic_rrf 
                }
                hybrid_fusion[doc_id] = res
            else:
                hybrid_fusion[doc_id]['semantic_rank'] = semantic_rrf_ranks[i]
                old_score = hybrid_fusion[doc_id]['rrf_score']
                new = old_score + semantic_rrf
                hybrid_fusion[doc_id]['rrf_score'] = new

        sorted_results = sorted(hybrid_fusion.items() , key = lambda item:item[1]['rrf_score'], reverse = True)
        
        return_res = []
        for id , map in sorted_results:
            res = {
                "id" : id,
                "title" : map['doc']['title'],
                "bm25_rank" : map['bm25_rank'],
                "semantic_rank" : map['semantic_rank'],
                "rrf_score" : round(float(map['rrf_score']) ,3),
                "description" : map['doc']['description']
            }
            return_res.append(res)
        
        return return_res[:limit]

def minmax_normalize(scores:list) -> list[float]:
    res = []
    
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        for i in range(len(scores)):
            res.append(1.0)
    else:
        for score in scores:
            res.append((score - min_score) / (max_score - min_score))
         
        
    return res

def hybrid_score(bm25_score:float , semantic_score:float , alpha:float = 0.5) -> float:
    return alpha * bm25_score + (1-alpha) * semantic_score

def rrf_score(rank:int , k:int = 60) -> float:
    return 1 / (k+rank)