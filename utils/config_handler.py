import yaml
from utils.path_tool import get_project_root, get_abs_path

def load_rag_config(config_path: str = get_abs_path("config/rag_config.yaml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
    