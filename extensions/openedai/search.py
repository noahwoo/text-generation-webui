import requests
from serpapi import GoogleSearch
import os
import chromadb
import pandas as pd
from collections import defaultdict 
import chromadb
from extensions.openedai.utils import Utils, OpenAIUtils
import ast
from scipy import spatial

os.environ['SERPAPI_API_KEY'] = '5ca1f5544dd75a93725b3017b9870c502c364be13f2610a3c2b83cdd5126487c'

'''
A class for search result of paper, include title, authors, abstract, pdf link, and publish information
'''

class Paper :
    def __init__(self, title, authors, summary, pdf_url, article_url, venue = None, affiliation = None) :
        self.title = title
        self.authors = authors
        self.summary = summary
        self.pdf_url = pdf_url
        self.article_url = article_url
        self.published = f"20{self.pdf_url.split('/')[-1].split('.')[0]}" if self.pdf_url is not None else 'NA'
        self.venue = venue
        self.affiliation = affiliation

    def local_pdf_path(self, target_dir) :
        return f"{target_dir}/{self.published}-{self.title}.pdf"
    
    @staticmethod
    def download_pdf_to_path(url, path) :
        with open(path, "wb") as f: 
            response = requests.get(url)
            f.write(response.content)
        return path

    def download_pdf(self, target_dir) :
        if self.pdf_url is None :
            return
        pdf_path = self.local_pdf_path(target_dir)
        return Paper.download_pdf_to_path(self.pdf_url, pdf_path)
        
class ScholarSearch :
    def __init__(self) :
        self.params = {
            "engine": "google_scholar",
            "api_key": os.environ['SERPAPI_API_KEY']
        }

    def search(self, query, site = None, num = None, year_from = None, author = None, affiliation = None) :

        if site is not None:
            query = f'{query} site:{site}'
        
        self.params.update({"q": f"{query}"})
        if year_from is not None :
            self.params.update({"as_ylo" : year_from})
        if num is not None :
            self.params.update({'num' : num})

        print(f"search params: {self.params}")
        search = GoogleSearch(self.params)
        results = search.get_dict()
        print(f"results: {results}")
        
        paper_list = []
        if 'organic_results' not in results :
            return paper_list
        
        organic_results = results["organic_results"]
        for result in organic_results :
            print(f"#{result['position']}: {result['title']}")
            print(f"Link: {result['link']}")
            paper_list.append(
                Paper(
                    result['title'],
                    ', '.join(a['name'] for a in result['publication_info']['authors']) if 'authors' in result['publication_info'] else None,
                    result['snippet'],
                    result['resources'][0]['link'] if 'resources' in result 
                                                and 'file_format' in result['resources'][0] 
                                                and result['resources'][0]['file_format'] == 'PDF' else None,
                    result['link'],
                )
            )
        return paper_list

'''
Repository for papers, 
'''
class PaperRepository :
    def __init__(self) :
        # fixed schema for PaperIndex
        self.schema = ["title", "authors", "published", "article_url", "pdf_url", "filepath", "embedding"]
        self.url2meta = defaultdict(dict)
        self.library_df = pd.DataFrame(list())
        pass
    
    def load(self, path) :
        # read csv
        self.library_df = pd.read_csv(path).reset_index()
        self.library_df.columns = self.schema
        self.library_df["embedding"] = self.library_df["embedding"].apply(ast.literal_eval) # eval emb vector
        # load url2meta dictionary
        for i, row in self.library_df.iterrows() :
            self.url2meta[row['pdf_url']].update(
                {
                    "filepath" : row["filepath"], 
                    "title": row["title"], 
                    "authors" : row["authors"], 
                    "published" : row["published"], 
                    "article_url" : row["article_url"]
                }
            )
        return True
    
    def strings_ranked_by_relatedness(self, 
        query: str,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 10,
        rel_th : float = 0.6) -> list[str]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = OpenAIUtils.embedding_request(query)
        query_embedding = query_embedding_response["data"][0]["embedding"]
        strings_and_relatednesses = [
            ({
                "filepath" : row["filepath"], 
                "title": row["title"], 
                "authors" : row["authors"], 
                "published" : row["published"], 
                "article_url" : row["article_url"], 
                "pdf_url" : row["pdf_url"]
            }, 
            relatedness_fn(query_embedding, row["embedding"]))
            for i, row in self.library_df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        print(f"Top {top_n} papers with relatedness to [{query}]:\n")
        for idx, string in enumerate(strings[:top_n]):
            print(f"{idx+1}. [{relatednesses[idx]}] [{string['title']}]")
        if relatednesses[top_n-1] > rel_th :
            print(f"Find paper with relatedness {relatednesses[top_n-1]} > {rel_th}!\n")
            return strings[:top_n]
        return None

    def create(self, path) :
        self.library_df.to_csv(path)

    def get_filepath(self, pdf_url) :
        assert self.library_df is not None, f"No index loaded"
        return self.url2meta[pdf_url]['filepath']
    
    def get_title(self, pdf_url) :
        assert self.library_df is not None, f"No index loaded"
        return self.url2meta[pdf_url]['title']
    
    def get_meta(self, pdf_url) :
        assert self.library_df is not None, f"No index loaded"
        return self.url2meta[pdf_url]
        
# chroma based local vector index, for paper trunks embedding
class VectorStore :
    def __init__(self) :
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("all-my-documents")
        pass
    
    def indexed(self, pdf_url) :
        return self.collection.get(f"{pdf_url}:0") is not None
        
    def add_paper(self, pdf_url, pdf_file_path, meta) :
        # check existence of local pdf_file
        assert os.path.exists(pdf_file_path), f"File not found at {pdf_file_path}"

        pdf_text = Utils.read_pdf(pdf_file_path)
        segments = Utils.chrunk_sentence(pdf_text)

        # chrunk text
        self.collection.add(
            embeddings=[OpenAIUtils.embedding_request(seg) for seg in segments], # too many requests for embedding
            documents=segments,
            metadatas=[meta] * len(segments), # for filtering
            ids=[f"{pdf_url}:{i}" for i in range(len(segments))]
        )
    
    def query(self, query_text = "", n_result = 10) :
        results = self.collection.query(
            query_embeddings=[OpenAIUtils.embedding_request(query_text)],
            query_texts=[query_text],
            n_results=n_result,
            # where={"metadata_field": "is_equal_to_this"}, # optional filter
            # where_document={"$contains":"search_string"}  # optional filter
        )
        return results
