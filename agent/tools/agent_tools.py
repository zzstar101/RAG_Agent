import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, TypeVar
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from langchain_core.tools import tool
from RAG.RAG_service import RAGSummarizeService
from uapi import UapiClient
from uapi.errors import UapiError


T = TypeVar("T")
API_BASE_URL = "https://uapis.cn"
API_TIMEOUT_SECONDS = 8
API_MAX_RETRIES = 3
API_RETRY_INTERVAL_SECONDS = 0.6

_uapi_client: UapiClient | None = None

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]  # 示例用户ID列表
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]
external_data = {}

rag = RAGSummarizeService()


def get_uapi_client() -> UapiClient:
    global _uapi_client

    if _uapi_client is None:
        _uapi_client = UapiClient(API_BASE_URL)

    return _uapi_client


def call_with_retry(action_name: str, operation: Callable[[], T]) -> T:
    last_error: Exception | None = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(operation).result(timeout=API_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            last_error = TimeoutError(f"{action_name}请求超时（>{API_TIMEOUT_SECONDS}s）")
            logger.warning(
                f"[外部API]{action_name}失败，第{attempt}/{API_MAX_RETRIES}次重试："
                f"{type(last_error).__name__}: {last_error}"
            )
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_INTERVAL_SECONDS * attempt)
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[外部API]{action_name}失败，第{attempt}/{API_MAX_RETRIES}次重试：{type(exc).__name__}: {exc}"
            )
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_INTERVAL_SECONDS * attempt)

    assert last_error is not None
    raise last_error

@tool(description="从向量存储中检索相关文档并生成摘要")
def rag_summarize(query: str) -> str:
    result = rag.rag_summarize(query)
    metrics = getattr(rag, "last_metrics", {})
    logger.debug(
        "[检索指标]hits=%s retrieval_ms=%.2f model_ms=%.2f",
        metrics.get("retrieval_hit_count", 0),
        float(metrics.get("retrieval_duration_ms", 0.0)),
        float(metrics.get("model_duration_ms", 0.0)),
    )
    return result

@tool(description="获取指定城市的天气信息，以消息字符串形式返回")
def get_weather(city: str) -> str:
    client = get_uapi_client()

    try:
        result = call_with_retry(
            action_name="天气查询",
            operation=lambda: client.misc.get_misc_weather(
                city=city,
                adcode="",
                extended=True,
                forecast=False,
                hourly=False,
                minutely=False,
                indices=False,
                lang="zh",
            ),
        )

        return (
            f"{result.get('city', city)}的天气：{result.get('weather', '未知')}，"
            f"温度：{result.get('temperature', '未知')}°C，"
            f"湿度：{result.get('humidity', '未知')}%，"
            f"风向：{result.get('wind_direction', '未知')}，"
            f"风力：{result.get('wind_power', '未知')}，"
            f"AQI指数：{result.get('aqi', '未知')}"
        )
    except (UapiError, TimeoutError, Exception) as exc:
        logger.error(f"[外部API]天气查询最终失败：{type(exc).__name__}: {exc}")
        return "天气服务暂时不可用，请稍后重试（可恢复错误）"
    
@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    client = get_uapi_client()

    try:
        result = call_with_retry(
            action_name="IP定位",
            operation=lambda: client.network.get_network_myip(
                source="commercial",
            ),
        )

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
    except (UapiError, TimeoutError, Exception) as exc:
        logger.error(f"[外部API]IP定位最终失败：{type(exc).__name__}: {exc}")
        return "未知城市（定位服务暂时不可用，可恢复错误）"
    
    
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
        
    
    
def _build_external_data_response(
    *,
    status: str,
    user_id: str,
    month: str,
    data: dict[str, str] | None,
    message: str,
) -> str:
    return json.dumps(
        {
            "status": status,
            "user_id": user_id,
            "month": month,
            "data": data,
            "message": message,
        },
        ensure_ascii=False,
    )


@tool(description="从外部系统中获取用户的使用记录，返回JSON字符串；status=ok时data为结构化记录，status=no_data时data为null，status=error时message说明错误")
def fetch_external_data(user_id: str, month: str) -> str:
    try:
        generate_external_data()

        record = external_data.get(user_id, {}).get(month)
        if record is None:
            logger.warning(f"[获取外部数据]未找到用户{user_id}在{month}的使用记录")
            return _build_external_data_response(
                status="no_data",
                user_id=user_id,
                month=month,
                data=None,
                message="未找到指定用户在指定月份的使用记录",
            )

        return _build_external_data_response(
            status="ok",
            user_id=user_id,
            month=month,
            data=record,
            message="获取成功",
        )
    except Exception as exc:
        logger.error(f"[获取外部数据]用户{user_id}在{month}的使用记录获取失败：{type(exc).__name__}: {exc}")
        return _build_external_data_response(
            status="error",
            user_id=user_id,
            month=month,
            data=None,
            message=f"{type(exc).__name__}: {exc}",
        )


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    return "动态注入上下文工具已调用"
