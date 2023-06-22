from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from pydantic import BaseModel
from langchain.utilities import BingSearchAPIWrapper
from paperqa import Docs
import requests
import re

class PaperQAWrapper(BaseModel):
    docs: Docs
    explanation: dict = {'ai' : [], 'human' : []}
    nr_iter : int = 1
    entire_output : str = ''

    class Config:
        arbitrary_types_allowed = True
        
    def _get_metadata_id(self, result):
        try:
            page_metadata = f"Title: {result['title']} - Link: {result['link']}"
            page_id = re.sub(r'[^\w\d]', '_', result["title"])
        except Exception as e:
            print(e)
            return None
        return page_metadata, page_id
    
     
    def _generate_search_query(self, query: str):
        """
        Generate a search query that can be used to find relevant pages in the web using BingSearchAPI to answer the question.
        """
        llm = OpenAI(temperature=0.2)
        template = """Here is a question:
            {query}
            Generate a search query that can be used to find relevant pages in the web using BingSearchAPI to answer the question.
            Do not mention 'research paper', 'pubmed', 'research', 'Abstract', 'Arxiv' or similar terms in the search query.
            Put quotation marks around relevant keywords and join them with ANDs.Do not generate more then 4 keywords.\n\n"""
        field_template = PromptTemplate(input_variables=["query"], template=template)
        field_chain = LLMChain(llm=llm, prompt=field_template)
        search_query = field_chain.run(query)
        search_query = search_query.replace('"', "").strip()
        return search_query
    
    def _bing_search(self, search_query: str):
        search = BingSearchAPIWrapper()
        results = search.results(search_query, 3)
        return_msg = ''

        for result in results:
            page_metadata, page_id = self._get_metadata_id(result)

            try:
                response = requests.get(result['link'])
            except Exception as e:
                return_msg += f'Failed extracting/writing the content of Page {page_id}. Error Message: {e}'

            if response:
                html_content = response.content
                file_full_path = '../documents/BingSearch_DBStrategyQA/' + page_id + '.html'
                with open(file_full_path, "wb") as f:
                    f.write(html_content)

                try:
                    self.docs.add(file_full_path, citation=result['link'], key=page_id)
                    return_msg += f'Found and Indexed Page {page_id}\n'

                except Exception as e:
                    return_msg += f"Found but failed to Index Page: {page_id}. Error Message: {e}"

        return return_msg
        
            
    
    def run(self, subquestion: str) -> str:
        #print(f'BEGIN: {len(self.docs.docs.keys())}')

        subquestion_feedback = input('Do you want to provide a better question for this step in order to change the reasoning path? If yes, please provide the question. If no, please type "n":')
        if subquestion_feedback != 'n':
            subquestion = subquestion_feedback

        while(True):
            # generate a search query that can be used to find relevant pages in the web using BingSearchAPI to answer the question.
            search_query = self._generate_search_query(subquestion)
            print(f"Generated Search Query : {search_query}\n")

            # search the web using BingSearchAPI, download and index the relevant pages
            return_msg = self._bing_search(search_query)
            print(return_msg)

            # get the answer to the question based on the most relevant indexed documents
            ai_answer = self.docs.query(subquestion, k=10, max_sources=4)
            print('\nAnswer:\n', ai_answer.answer, '\n\n')

            if 'insufficient information' in ai_answer.answer.lower() or 'cannot answer' in ai_answer.answer.lower():
                subquestion = input('Insufficient information/context to answer the question. Please provide another sub-question for this step!')
            else:
                break
      
        print(f'Supporting evidence documents: \n')
        print(f'Nr of supporting evidence documents: {len(ai_answer.contexts)}\n')
        # get the top 2 most relevant supporting evidence documents
        top2_rel_docs = ai_answer.contexts[:2]
        for nr, rel_doc_summary in enumerate(top2_rel_docs):
            summarized_doc = rel_doc_summary.context
            print(f'Doc {nr+1}:\n{summarized_doc}\n')

        #doc_urls = [el.strip() for el in ai_answer.references.split("Link:") if 'https' in el.strip()]
        print(ai_answer.references)
        print(f'Document URLs : {ai_answer.references.split(":")[1].strip()}\n\n')

        print('Rate this explanation step on a scale of 1 to 5 (1 - not reasonable, 5 - very reasonable)')
        rating = int(input('Is this step reasonable? (1 to 5)'))

        subanswer_feedback = input('Do you want to suggest a better sub-answer/alternative for this step? If yes, please provide the sub-answer. If no, please type "n":')    
        if subanswer_feedback != 'n':
            explanation_step = {f'Subquestion #{self.nr_iter}' : subquestion,
                                f'Subanswer #{self.nr_iter}' :  subanswer_feedback, 
                                f'Rating #{self.nr_iter}' : 5
                                }
            subanswer = subanswer_feedback
            self.explanation['human'].append(explanation_step)
        else:
            explanation_step = {f'Subquestion #{self.nr_iter}' : subquestion,
                            f'Subanswer #{self.nr_iter}' :  ai_answer.answer, 
                            f'Rating #{self.nr_iter}' : rating}
            
            self.explanation['ai'].append(explanation_step)    
            self.explanation['human'].append(explanation_step)       
            subanswer = ai_answer.answer
        
        step_info = f'Step {self.nr_iter}\nQ: {subquestion}\nA: {ai_answer.answer}\n\n'
        self.entire_output += step_info
        self.nr_iter += 1
        return subanswer

           

class PaperQATool(BaseTool):
    name = "Intermediate Answer"
    description = """This tool is used to generate a search query for a given question, find relevant documents in the web using BingSearchAPI, download them and add to the library and finally generate an answer to the question. The input is a natural language question."""
    api_wrapper: PaperQAWrapper = None

    def __init__(self, docs: Docs):
        api_wrapper = PaperQAWrapper(docs=docs)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        res = self.api_wrapper.run(query)
        return res

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError