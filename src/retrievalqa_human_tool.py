from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import VectorStore
from langchain.tools.base import BaseTool
from pydantic import BaseModel


class HumanQAWrapper(BaseModel):
    db: VectorStore

    class Config:
        arbitrary_types_allowed = True

    def run(self, query: str) -> str:
        
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.db.as_retriever(search_kwargs={'k' : 2}), return_source_documents=True)
        result = qa_chain({"query": query})
        print('\nAnswer:\n', result['result'], '\n\n')
        for source_nr, source in enumerate(result['source_documents']):
            print(f'Doc {source_nr + 1}:\n{source.page_content} \n\n')

        user_feedback = input('Do you want to suggest a better alternative for this step? (y/n)')
        while(True):
            if user_feedback.lower() == 'y':
                final_answer = input('Please provide your answer:')
                break
            elif user_feedback.lower() == 'n':
                final_answer = result['result']
                break
            else:
                print('Please provide a valid input. Input should be either "y" or "n". Try again!')
                user_feedback = input('Do you want to suggest a better alternative for this step? (y/n)')
        return final_answer

class HumanQATool(BaseTool):
    name = "Intermediate Answer"
    description = """This tool is useful to ask the user for input and for answering questions.
    In principle, it can also fail, telling you that there is insufficient information, or by saying that there is no
    clear answer. Input should be a question."""
    api_wrapper: HumanQAWrapper = None

    def __init__(self, db: VectorStore):
        api_wrapper = HumanQAWrapper(db=db)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError