from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from utils.path_tool import get_abs_path


class ConfigValidationError(ValueError):
    pass


def _load_yaml_file(config_path: str, encoding: str = "utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigValidationError(f"配置文件必须返回一个映射对象: {config_path}")

    return data


def _merge_dict(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value

    return merged


def _load_config(config_path: str, encoding: str = "utf-8") -> dict:
    base_path = Path(config_path)
    config_data = _load_yaml_file(str(base_path), encoding=encoding)

    env_name = os.getenv("RAG_AGENT_ENV") or os.getenv("APP_ENV") or "default"
    env_name = env_name.strip().lower()
    if env_name and env_name != "default":
        env_path = base_path.with_name(f"{base_path.stem}.{env_name}{base_path.suffix}")
        if env_path.exists():
            config_data = _merge_dict(config_data, _load_yaml_file(str(env_path), encoding=encoding))

    return config_data


def _require_string(config: dict, key: str, *, source: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{source} 缺少有效的 {key} 配置")
    return value.strip()


def _require_int(config: dict, key: str, *, source: str, minimum: int | None = None) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ConfigValidationError(f"{source} 的 {key} 必须是整数")
    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"{source} 的 {key} 必须大于等于 {minimum}")
    return value


def _require_float(
    config: dict,
    key: str,
    *,
    source: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{source} 的 {key} 必须是数字")
    numeric_value = float(value)
    if minimum is not None and numeric_value < minimum:
        raise ConfigValidationError(f"{source} 的 {key} 必须大于等于 {minimum}")
    if maximum is not None and numeric_value > maximum:
        raise ConfigValidationError(f"{source} 的 {key} 必须小于等于 {maximum}")
    return numeric_value


def _require_string_list(config: dict, key: str, *, source: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f"{source} 的 {key} 必须是非空字符串列表")

    normalized_value: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigValidationError(f"{source} 的 {key} 包含非法项")

        if key == "separators":
            normalized_value.append(item)
            continue

        if not item.strip():
            raise ConfigValidationError(f"{source} 的 {key} 包含非法项")

        normalized_value.append(item.strip())

    return normalized_value


def _normalize_chroma_config(config: dict) -> dict:
    normalized = copy.deepcopy(config)

    def _coerce_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    if "persist_directory" not in normalized and "persist_dictionary" in normalized:
        normalized["persist_directory"] = normalized["persist_dictionary"]
    if "persist_dictionary" not in normalized and "persist_directory" in normalized:
        normalized["persist_dictionary"] = normalized["persist_directory"]

    if "chunks_size" not in normalized and "chunk_size" in normalized:
        normalized["chunks_size"] = normalized["chunk_size"]
    if "chunk_size" not in normalized and "chunks_size" in normalized:
        normalized["chunk_size"] = normalized["chunks_size"]

    if "chunks_overlap" not in normalized and "chunk_overlap" in normalized:
        normalized["chunks_overlap"] = normalized["chunk_overlap"]
    if "chunk_overlap" not in normalized and "chunks_overlap" in normalized:
        normalized["chunk_overlap"] = normalized["chunks_overlap"]

    if "allow_knowledge_file_types" not in normalized and "allow_knowledge_file_type" in normalized:
        normalized["allow_knowledge_file_types"] = normalized["allow_knowledge_file_type"]

    normalized.setdefault("retrieval_mode", "rerank")
    top_k = _coerce_int(normalized.get("k"), 3)
    normalized.setdefault("fetch_k", max(top_k * 2, top_k))
    normalized.setdefault("lambda_mult", 0.5)
    normalized.setdefault("score_threshold", 0.0)

    return normalized


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    config = _load_config(config_path, encoding=encoding)
    source = f"RAG配置文件 {config_path}"
    _require_string(config, "chat_model_name", source=source)
    _require_string(config, "embedding_model_name", source=source)
    return config


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    config = _normalize_chroma_config(_load_config(config_path, encoding=encoding))
    source = f"Chroma配置文件 {config_path}"
    _require_string(config, "collection_name", source=source)
    _require_string(config, "persist_dictionary", source=source)
    _require_int(config, "k", source=source, minimum=1)
    _require_string(config, "data_path", source=source)
    _require_string(config, "md5_hex_store", source=source)
    _require_string_list(config, "allow_knowledge_file_types", source=source)
    chunk_size = _require_int(config, "chunks_size", source=source, minimum=1)
    chunk_overlap = _require_int(config, "chunks_overlap", source=source, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigValidationError(f"{source} 的 chunks_overlap 必须小于 chunks_size")
    _require_string_list(config, "separators", source=source)
    _require_string(config, "retrieval_mode", source=source)
    _require_int(config, "fetch_k", source=source, minimum=1)
    _require_float(config, "lambda_mult", source=source, minimum=0.0, maximum=1.0)
    _require_float(config, "score_threshold", source=source, minimum=0.0, maximum=1.0)
    return config


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    config = _load_config(config_path, encoding=encoding)
    source = f"提示词配置文件 {config_path}"
    _require_string(config, "main_prompt_path", source=source)
    _require_string(config, "rag_summarization_prompt_path", source=source)
    _require_string(config, "report_prompt_path", source=source)
    return config


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    config = _load_config(config_path, encoding=encoding)
    source = f"Agent配置文件 {config_path}"
    _require_string(config, "external_data_path", source=source)
    return config


rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()

