import hashlib
import os
from logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

def get_file_md5_hex(filepath: str):
    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算]文件{filepath}不存在")
        return 
    
    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算]路径{filepath}不是一个文件")
        return
    
    md5_hash = hashlib.md5()
    
    chunk_size = 4096  # 4KB
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
            md5_hex = md5_hash.hexdigest()
            return md5_hex
        
    except Exception as e:
        logger.error(f"[MD5计算]读取文件{filepath}时发生错误: {str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    
    if not os.path.isdir(path):
        logger.error(f"[文件列表]路径{path}不是一个目录")
        return allowed_types
    
    for f in os.listdir(path):
        if f.lower().endswith(allowed_types):
            files.append(os.path.join(path, f))
            
        return tuple(files)


def pdf_loader(filepath: str, password: str | None = None) -> list[Document]:
    return PyPDFLoader(filepath, password=password).load()
    

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath).load()