from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_config

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass
    
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[ChatTongyi|DashScopeEmbeddings]:
        return ChatTongyi(model=rag_config["chat_model_name"])
    
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[ChatTongyi|DashScopeEmbeddings]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])
    
    
chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
        
    