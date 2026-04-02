from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedding_model
from utils.config_handler import chroma_config
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float | None
    confidence: str


class Md5IndexCache:
    def __init__(self, store_path: str):
        self.store_path = Path(get_abs_path(store_path))
        self._hashes = self._load_hashes()

    def _load_hashes(self) -> set[str]:
        if not self.store_path.exists():
            return set()

        with open(self.store_path, "r", encoding="utf-8") as file_handle:
            return {line.strip() for line in file_handle if line.strip()}

    def contains(self, md5_hex: str) -> bool:
        return md5_hex in self._hashes

    def add(self, md5_hex: str) -> None:
        if md5_hex in self._hashes:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(md5_hex + "\n")
        self._hashes.add(md5_hex)


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model,
            persist_directory=get_abs_path(chroma_config["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunks_size"],
            chunk_overlap=chroma_config["chunks_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )
        self.md5_cache = Md5IndexCache(chroma_config["md5_hex_store"])
        self.top_k = chroma_config["k"]
        self.fetch_k = max(chroma_config.get("fetch_k", self.top_k), self.top_k)
        self.retrieval_mode = str(chroma_config.get("retrieval_mode", "rerank")).lower()
        self.lambda_mult = float(chroma_config.get("lambda_mult", 0.5))
        self.score_threshold = float(chroma_config.get("score_threshold", 0.0))

    def get_retriever(self):
        search_kwargs = {"k": self.top_k}
        search_type = "similarity"

        if self.retrieval_mode == "mmr":
            search_type = "mmr"
            search_kwargs["fetch_k"] = self.fetch_k
            search_kwargs["lambda_mult"] = self.lambda_mult

        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def _extract_terms(self, text: str) -> set[str]:
        terms = set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
        return {term for term in terms if term}

    def _rerank_documents(
        self,
        query: str,
        scored_documents: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        if self.retrieval_mode != "rerank":
            return scored_documents

        query_terms = self._extract_terms(query)
        if not query_terms:
            return scored_documents

        reranked_documents: list[tuple[Document, float]] = []
        for document, base_score in scored_documents:
            metadata_text = " ".join(f"{key}:{value}" for key, value in document.metadata.items())
            candidate_terms = self._extract_terms(f"{document.page_content} {metadata_text}")
            overlap_score = len(query_terms & candidate_terms) / max(len(query_terms), 1)
            combined_score = (float(base_score) * 0.7) + (overlap_score * 0.3)
            reranked_documents.append((document, combined_score))

        reranked_documents.sort(key=lambda item: item[1], reverse=True)
        return reranked_documents

    def _confidence_label(self, score: float | None, rank: int) -> str:
        if score is None:
            return "high" if rank == 1 else "medium" if rank == 2 else "low"

        if score >= 0.8:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"

    def retrieve_documents(self, query: str) -> list[RetrievedDocument]:
        fetch_k = max(self.fetch_k, self.top_k)
        scored_documents = self.vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
        scored_documents = self._rerank_documents(query, scored_documents)

        filtered_documents = [
            RetrievedDocument(
                document=document,
                score=score,
                confidence=self._confidence_label(score, index + 1),
            )
            for index, (document, score) in enumerate(scored_documents)
            if score >= self.score_threshold
        ]

        return filtered_documents[: self.top_k]

    def load_document(self):
        def get_file_documents(read_path: str):
            lowered_path = read_path.lower()
            if lowered_path.endswith(".txt") or lowered_path.endswith(".md"):
                return txt_loader(read_path)

            if lowered_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allowed_file_path = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_types"]),
        )

        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.warning(f"[加载知识库]文件 {path} 无法计算 MD5，跳过")
                continue

            if self.md5_cache.contains(md5_hex):
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
                self.md5_cache.add(md5_hex)
                logger.info(f"[加载知识库]文件 {path} 已经成功加载到知识库中")
            except Exception as e:
                logger.error(f"[加载知识库]文件 {path} 加载失败，错误信息: {str(e)}", exc_info=True)


if __name__ == "__main__":
    vector_store_storage = VectorStoreService()
    vector_store_storage.load_document()
    retriever = vector_store_storage.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)