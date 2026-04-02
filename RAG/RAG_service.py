from RAG.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from model.factory import chat_model




class RAGSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()  # Initialize your vector store here
        self.retriever = self.vector_store.get_retriever()  # Initialize your retriever here
        self.prompt_text = load_rag_prompt()  # Initialize your prompt text here
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)  # Initialize your prompt template here
        self.model = chat_model  # Initialize your model here
        self.chain = self.__init__chain()  # Initialize your chain here
        
    def __init__chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)
    
    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            
        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )
            
            