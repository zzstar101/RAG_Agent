from abc import ABC, abstractmethod
import os
import socket
from typing import Generic, TypeVar
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_config
from utils.logger_handler import logger

TModel = TypeVar("TModel")


class BaseModelFactory(ABC, Generic[TModel]):
    @abstractmethod
    def generator(self) -> TModel:
        raise NotImplementedError


class ChatModelFactory(BaseModelFactory[BaseChatModel]):
    def generator(self) -> BaseChatModel:
        model_name = str(rag_config.get("chat_model_name", "")).strip()
        return ChatTongyi(model=model_name)


class EmbeddingsFactory(BaseModelFactory[Embeddings]):
    def generator(self) -> Embeddings:
        model_name = str(rag_config.get("embedding_model_name", "")).strip()
        return DashScopeEmbeddings(model=model_name)


def run_startup_checks(
    *,
    connectivity_host: str = "dashscope.aliyuncs.com",
    connectivity_port: int = 443,
    timeout_seconds: float = 2.0,
) -> dict[str, dict[str, str | bool]]:
    chat_model_name = str(rag_config.get("chat_model_name", "")).strip()
    embedding_model_name = str(rag_config.get("embedding_model_name", "")).strip()
    has_model_name = bool(chat_model_name and embedding_model_name)

    has_credential = bool(os.getenv("DASHSCOPE_API_KEY", "").strip())

    connectivity_ok = False
    connectivity_detail = ""
    try:
        with socket.create_connection((connectivity_host, connectivity_port), timeout=timeout_seconds):
            connectivity_ok = True
            connectivity_detail = "ok"
    except OSError as exc:
        connectivity_detail = f"{type(exc).__name__}: {exc}"

    return {
        "model_name": {
            "ok": has_model_name,
            "detail": f"chat={chat_model_name or '-'}, embedding={embedding_model_name or '-'}",
        },
        "credential": {
            "ok": has_credential,
            "detail": "DASHSCOPE_API_KEY configured" if has_credential else "DASHSCOPE_API_KEY missing",
        },
        "connectivity": {
            "ok": connectivity_ok,
            "detail": connectivity_detail,
        },
    }


def _log_startup_check_summary() -> None:
    checks = run_startup_checks()
    for check_name, item in checks.items():
        if item["ok"]:
            logger.info("[启动自检]%s 通过：%s", check_name, item["detail"])
        else:
            logger.warning("[启动自检]%s 失败：%s", check_name, item["detail"])
    
_log_startup_check_summary()

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()

