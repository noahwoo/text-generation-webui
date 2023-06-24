import arxiv
import requests
from serpapi import GoogleSearch
import os
from unittest import result

os.environ['SERPAPI_API_KEY'] = '5ca1f5544dd75a93725b3017b9870c502c364be13f2610a3c2b83cdd5126487c'

'''
A class for search result of paper, include title, authors, abstract, pdf link, and publish information
'''

class Result :
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
        return Result.download_pdf_to_path(self.pdf_url, pdf_path)
        
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
                Result(
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
