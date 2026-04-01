from config_handler import prompts_config
from path_tool import get_abs_path
from logger_handler import logger

def load_system_prompt():
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
        
    except KeyError as e:
        logger.error(f"[加载系统提示词]配置文件缺少main_prompt_path字段")
        raise e
    
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[加载系统提示词]读取文件{system_prompt_path}时发生错误: {str(e)}")
        raise e
    
def load_rag_prompt():
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarization_prompt_path"])
        
    except KeyError as e:
        logger.error(f"[加载RAG提示词]配置文件缺少rag_summarization_prompt_path字段")
        raise e
    
    try:
        with open(rag_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[加载RAG提示词]读取文件{rag_prompt_path}时发生错误: {str(e)}")
        raise e
    
def load_report_prompt():
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
        
    except KeyError as e:
        logger.error(f"[加载报告提示词]配置文件缺少report_prompt_path字段")
        raise e
    
    try:
        with open(report_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[加载报告提示词]读取文件{report_prompt_path}时发生错误: {str(e)}")
        raise e
    
