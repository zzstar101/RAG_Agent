import os
import random
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from langchain_core.tools import tool
from RAG.RAG_service import RAGSummarizeService
from uapi import UapiClient
from uapi.errors import UapiError


user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]  # 示例用户ID列表
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]
external_data = {}

rag = RAGSummarizeService()

@tool(description="从向量存储中检索相关文档并生成摘要")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气信息，以消息字符串形式返回")
def get_weather(city: str) -> str:
    client = UapiClient("https://uapis.cn")
    try:
        result = client.misc.get_misc_weather(city=city, adcode="", extended=True, forecast=False, hourly=False, minutely=False, indices=False, lang="zh")
        return f"{result['city']}的天气：{result['weather']}，温度：{result['temperature']}°C，湿度：{result['humidity']}%，风向：{result['wind_direction']}，风力：{result['wind_power']}，AQI指数：{result['aqi']}"
    except UapiError as exc:
        return f"API error: {exc}"
    
@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    client = UapiClient("https://uapis.cn")
    try:
        result = client.network.get_network_myip(source="commercial")

        # 优先使用接口直出城市字段
        city = result.get("city")
        if isinstance(city, str) and city.strip():
            return city.strip()

        # 回退：从 region 中提取城市
        region = result.get("region", "")
        if isinstance(region, str) and region.strip():
            parts = [p for p in region.split() if p]
            if len(parts) >= 3 and parts[0] == "中国":
                return parts[2]
            if len(parts) >= 2:
                return parts[1]
            return parts[0]

        return "未知城市"
    except UapiError as exc:
        return f"API error: {exc}"
    
    
@tool(description="获取用户ID,以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份,以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)
    
def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # 跳过标题行
                arr: list[str] = line.strip().split(",")
                
                user_id = arr[0].replace('"', '')
                feature = arr[1].replace('"', '')
                efficiency = arr[2].replace('"', '')
                consumables = arr[3].replace('"', '')
                comparison = arr[4].replace('"', '')
                time = arr[5].replace('"', '')
                
                if user_id not in external_data:
                    external_data[user_id] = {}
                    
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison
                }
        
    
    
@tool(description="从外部系统中获取用户的使用记录，以纯字符串形式返回，若未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[获取外部数据]未找到用户{user_id}在{month}的使用记录")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    return "动态注入上下文工具已调用"
