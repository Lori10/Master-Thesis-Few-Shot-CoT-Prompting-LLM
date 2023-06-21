from paperqa import Docs
import wikipedia
import requests
import re
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from pydantic import BaseModel


class WikiWrapper(BaseModel):
    docs: Docs

    class Config:
        arbitrary_types_allowed = True
        
    
    def _get_metadata_id(self, page):
        try:
            page_metadata = f"Title: {page.title} - Link: {page.url}"
            page_id = re.sub(r'[^\w\d]', '_', page.title)
        except Exception as e:
            print(e)
            return None
        return page_metadata, page_id   
    
    def _preprocess_page(self, page):
        page_metadata, page_id = self._get_metadata_id(page)

        try:
            response = requests.get(page.url)
        except Exception as e:
            print(e)
            print(f'Failed extracting content of Page: {page_metadata}')

        html_content = response.content
        file_full_path = 'Documents/Wikipedia/' + page_id + '.html'
        with open(file_full_path, "wb") as f:
            f.write(html_content)

        if file_full_path in self.docs.docs:
            print("This Page already in the Docs")
        else:
            print(f"Indexing Page : {page_id}")

            try:
                self.docs.add(file_full_path, citation=page_metadata)
            except Exception as e:
                print(e, type(e))
                print(f"Failed to index Page: {page_id}")
    
    
    
    def run(self, query: str) -> str:
        llm = OpenAI(temperature=0.2)
        template = """Here is a question:
            {query}
            Generate a search query that can be used to find relevant pages in wikipedia to answer the question.
            Do not mention 'wikipedia' or similar terms in the search query.
            Do not generate more than 2 keywords.\n\n"""
        field_template = PromptTemplate(input_variables=["query"], template=template)
        field_chain = LLMChain(llm=llm, prompt=field_template)
        search_query = field_chain.run(query)
        search_query = search_query.replace('"', "")
        print(f"Generated Search Query : {search_query}")
        
        page_names = wikipedia.search(search_query, results=3)
        print(f'Using Wikipedia Search Engine found these pages : {[page for page in page_names]}')
        return_msg = ''
        
        for page_name in page_names:
            try:
                page = wikipedia.page(page_name, auto_suggest=False)
                return_msg += f"Found Page {page.title}\n" 
            except wikipedia.exceptions.DisambiguationError as e:
                related_pages = []
                for i in range(len(e.options)):
                    related_pages.append(e.options[i])
                    try:
                        page = wikipedia.page(e.options[i], auto_suggest=False)
                    except:
                        print(f'Page {e.options[i]} does not exist')
                        continue
                    self._preprocess_page(page)
                return_msg += f"Found Related to the Page {e.title} these Pages {[el for el in related_pages]}\n"
                continue
                
            except wikipedia.exceptions.PageError as e:
                print(f'Page {page_name} does not exist!')   
                continue
            
            self._preprocess_page(page)


        return return_msg
    
    
class WikiTool(BaseTool):
    name = "WikiTool"
    description = """ This tool is used to generate a search query for a given question, find relevant documents in the wikipedia, extract them and add to the library. The input is a natural language question."""
    api_wrapper: WikiWrapper = None

    def __init__(self, docs: Docs):
        api_wrapper = WikiWrapper(docs=docs)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        res_bool = self.api_wrapper.run(query)
        return res_bool

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError