import os

from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config['collection_name'],
            embedding_function=embedding_model,
            persist_directory=chroma_config['persist_dictionary']
        )
        
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config['chunks_size'],
            chunk_overlap=chroma_config['chunks_overlap'],
            separators=chroma_config['separators'],
            length_function=len
        )
        
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config['k']})
    
    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_config['md5_hex_store'])):
                open(get_abs_path(chroma_config['md5_hex_store']), 'w', encoding='utf-8').close()
                return False
            
            with open(get_abs_path(chroma_config['md5_hex_store']), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                    
                return False
            
        def save_md5_hex(md5_to_save: str):
            with open(get_abs_path(chroma_config['md5_hex_store']), 'a', encoding='utf-8') as f:
                f.write(md5_to_save + '\n')
                
        def get_file_documents(read_path: str):
            if read_path.endswith('.txt'):
                return txt_loader(read_path)
            
            if read_path.endswith('.pdf'):
                return pdf_loader(read_path)
            
            return []
        
        allowed_file_path = listdir_with_allowed_type(
            get_abs_path(chroma_config['data_path']), 
            tuple(chroma_config['allow_knowledge_file_types'])
            )
        
        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]文件 {path} 已经存在于知识库中，跳过")
                continue
            
            try:
                documents: list[Document] = get_file_documents(path)
                
                if not documents:
                    logger.warning(f"[加载知识库]文件 {path} 无法被解析，跳过")
                    continue
                
                split_document: list[Document] = self.spliter.split_documents(documents)
                
                if not split_document:
                    logger.warning(f"[加载知识库]文件 {path} 无法被切分，跳过")
                    continue
                
                self.vector_store.add_documents(split_document)
                
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]文件 {path} 已经成功加载到知识库中")
            except Exception as e:
                logger.error(f"[加载知识库]文件 {path} 加载失败，错误信息: {str(e)}", exc_info=True)
                
if __name__ == "__main__":
    vector_store_storage = VectorStoreStorage()
    vector_store_storage.load_document()
    retriever = vector_store_storage.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)