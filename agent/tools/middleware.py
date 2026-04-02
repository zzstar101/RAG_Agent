from typing import Callable
from httpx import request
from langchain.agents.middleware import before_model, wrap_tool_call, AgentState, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompt, load_system_prompt

@wrap_tool_call
def monitor_tool(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage| Command]) -> ToolMessage| Command:
    logger.info(f"[工具调用监控]执行工具：{request.tool_call['name']}")
    logger.info(f"[工具调用监控]传入参数：{request.tool_call['args']}")
    
    try:
        result = handler(request)
        logger.info(f"[工具调用监控]工具{request.tool_call['name']}调用成功")
        
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
            
        return result
    except Exception as e:
        logger.error(f"[工具调用监控]工具{request.tool_call['name']}调用失败：{str(e)}")
        raise e

@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    logger.info(f"[模型调用监控]即将调用模型，附带{len(state['messages'])}条消息")
    
    logger.debug(f"[模型调用监控] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    isReport = request.runtime.context.get("report", False)
    if isReport:
        return load_report_prompt()
    
    return load_system_prompt()


    