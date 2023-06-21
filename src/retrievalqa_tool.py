from langchain.vectorstores.base import VectorStore
from langchain.tools.base import BaseTool
from pydantic import BaseModel
from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class RetrievalQAWrapper(BaseModel):
    db: VectorStore

    class Config:
        arbitrary_types_allowed = True

    def run(self, query: str) -> str:
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.db.as_retriever(search_kwargs={'k' : 2}), return_source_documents=True)
        result = qa_chain({"query": query})
        #return result['result']
        print('\nAnswer:\n', result['result'], '\n\n')
        qa_with_sources = VectorDBQAWithSourcesChain.from_chain_type(
                                                        llm=OpenAI(),
                                                        chain_type="stuff",
                                                        vectorstore=self.db)
        
        qa_with_sources.k = 10
        return f'\n{qa_with_sources({"question": query}, return_only_outputs=True)["answer"]}\n'
    
class RetrievalQATool(BaseTool):
    name = "Intermediate Answer"
    description = """This tool is useful for answering questions.
    In principle, it can also fail, telling you that there is insufficient information, or by saying that there is no
    clear answer. Input should be a question."""
    api_wrapper: RetrievalQAWrapper = None

    def __init__(self, db: VectorStore):
        api_wrapper = RetrievalQAWrapper(db=db)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError
    



    

