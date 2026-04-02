import hashlib
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger
from utils.path_tool import get_abs_path, is_path_within_project


def _resolve_workspace_path(filepath: str) -> Path:
    candidate = Path(filepath)
    if not candidate.is_absolute():
        candidate = Path(get_abs_path(filepath))

    resolved = candidate.resolve(strict=False)
    if not is_path_within_project(resolved):
        raise ValueError(f"path is outside project root: {filepath}")

    return resolved

def get_file_md5_hex(filepath: str):
    try:
        file_path = _resolve_workspace_path(filepath)
    except ValueError as exc:
        logger.error(f"[MD5计算]{exc}")
        return None

    if not file_path.is_file():
        logger.error(f"[MD5计算]文件{file_path}不存在或不是一个文件")
        return None

    md5_hash = hashlib.md5()
    chunk_size = 4096  # 4KB

    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    except Exception as e:
        logger.error(f"[MD5计算]读取文件{file_path}时发生错误: {str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []

    try:
        directory_path = _resolve_workspace_path(path)
    except ValueError as exc:
        logger.error(f"[文件列表]{exc}")
        return tuple()

    if not directory_path.is_dir():
        logger.error(f"[文件列表]路径{directory_path}不是一个目录")
        return tuple()

    normalized_types = tuple(file_type.lower() for file_type in allowed_types)

    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.name.lower().endswith(normalized_types):
            files.append(str(file_path.resolve(strict=False)))

    return tuple(files)


def pdf_loader(filepath: str, password: str | None = None) -> list[Document]:
    resolved_path = _resolve_workspace_path(filepath)
    return PyPDFLoader(str(resolved_path), password=password).load()
    

def txt_loader(filepath: str) -> list[Document]:
    resolved_path = _resolve_workspace_path(filepath)
    return TextLoader(str(resolved_path), encoding='utf-8').load()