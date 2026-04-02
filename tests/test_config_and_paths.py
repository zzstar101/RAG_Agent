import os
import tempfile
import unittest
from pathlib import Path

from RAG.vector_store import Md5IndexCache
from utils.config_handler import ConfigValidationError, load_chroma_config, load_rag_config
from utils.file_handler import listdir_with_allowed_type
from utils.path_tool import get_abs_path, get_project_root


class PathAndConfigTests(unittest.TestCase):
    def test_get_abs_path_rejects_escape(self):
        with self.assertRaises(ValueError):
            get_abs_path("../outside.txt")

    def test_listdir_with_allowed_type_filters_files(self):
        project_root = Path(get_project_root())
        with tempfile.TemporaryDirectory(dir=project_root) as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "a.txt").write_text("a", encoding="utf-8")
            (temp_path / "b.pdf").write_text("b", encoding="utf-8")
            (temp_path / "c.jpg").write_text("c", encoding="utf-8")

            result = listdir_with_allowed_type(str(temp_path.relative_to(project_root)), (".txt", ".pdf"))
            result_names = {Path(item).name for item in result}

            self.assertEqual(result_names, {"a.txt", "b.pdf"})

    def test_load_rag_config_uses_environment_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_path = temp_path / "rag.yml"
            overlay_path = temp_path / "rag.dev.yml"
            base_path.write_text(
                "chat_model_name: base-model\nembedding_model_name: base-embedding\n",
                encoding="utf-8",
            )
            overlay_path.write_text(
                "chat_model_name: dev-model\n",
                encoding="utf-8",
            )

            original_env = os.environ.get("RAG_AGENT_ENV")
            os.environ["RAG_AGENT_ENV"] = "dev"
            try:
                config = load_rag_config(str(base_path))
            finally:
                if original_env is None:
                    os.environ.pop("RAG_AGENT_ENV", None)
                else:
                    os.environ["RAG_AGENT_ENV"] = original_env

            self.assertEqual(config["chat_model_name"], "dev-model")
            self.assertEqual(config["embedding_model_name"], "base-embedding")

    def test_load_chroma_config_validates_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "chroma.yml"
            config_path.write_text(
                "\n".join(
                    [
                        "collection_name: agent",
                        "persist_dictionary: chroma",
                        "k: 0",
                        "data_path: data",
                        "md5_hex_store: md5.text",
                        "allow_knowledge_file_types: ['.txt']",
                        "chunks_size: 200",
                        "chunks_overlap: 20",
                        "separators: ['\\n']",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ConfigValidationError):
                load_chroma_config(str(config_path))

    def test_md5_index_cache_persists_entries(self):
        project_root = Path(get_project_root())
        with tempfile.TemporaryDirectory(dir=project_root) as temp_dir:
            temp_path = Path(temp_dir)
            relative_store_path = (temp_path / "md5_store.txt").relative_to(project_root).as_posix()

            cache = Md5IndexCache(relative_store_path)
            self.assertFalse(cache.contains("abc123"))

            cache.add("abc123")
            self.assertTrue(cache.contains("abc123"))

            cache_again = Md5IndexCache(relative_store_path)
            self.assertTrue(cache_again.contains("abc123"))


if __name__ == "__main__":
    unittest.main()