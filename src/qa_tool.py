from paperqa import Docs
from langchain.tools.base import BaseTool
from pydantic import BaseModel


class QAWrapper(BaseModel):
    docs: Docs

    class Config:
        arbitrary_types_allowed = True

    def run(self, query: str) -> str:
        # Use k text passages in up to max_sources different documents
        answer = self.docs.query(query, k=10, max_sources=5)
        return answer.formatted_answer
    
class QATool(BaseTool):
    name = "Question Answering"
    description = """This tool is useful for answering questions from the content of downloaded documents from the web.
    In principle, it can also fail, telling you that there is insufficient information, or by saying that there is no
    clear answer. Input should be a question."""
    api_wrapper: QAWrapper = None

    def __init__(self, docs: Docs):
        api_wrapper = QAWrapper(docs=docs)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError