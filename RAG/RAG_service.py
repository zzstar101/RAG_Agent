from RAG.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from model.factory import chat_model




class RAGSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__init__chain()
        
    def __init__chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query: str) -> list[Document]:
        retrieved_documents = self.vector_store.retrieve_documents(query)
        return [retrieved_document.document for retrieved_document in retrieved_documents]

    def _format_source_summary(self, retrieved_documents) -> str:
        if not retrieved_documents:
            return "\n\n参考来源：未命中知识库中的相关内容。"

        source_lines = ["\n\n参考来源："]
        for index, retrieved_document in enumerate(retrieved_documents, start=1):
            source = retrieved_document.document.metadata.get("source", "未知来源")
            score_text = f"{retrieved_document.score:.2f}" if retrieved_document.score is not None else "未知"
            source_lines.append(
                f"- [{index}] 来源：{source} | 置信度：{retrieved_document.confidence} | 相关度：{score_text}"
            )

        return "\n".join(source_lines)
    
    def rag_summarize(self, query: str) -> str:
        retrieved_documents = self.vector_store.retrieve_documents(query)
        context_docs = [retrieved_document.document for retrieved_document in retrieved_documents]
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
        ) + self._format_source_summary(retrieved_documents)
            
            