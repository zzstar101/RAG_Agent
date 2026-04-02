from typing import Callable, List
from langchain.agents.middleware import before_model, wrap_tool_call, AgentState, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from dataclasses import dataclass
import inspect
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompt, load_system_prompt

class MiddlewareRegistrationError(ValueError):
    """Raised when middleware registration is invalid."""

@dataclass(frozen=True)
class MiddlewareRegistration:
    name: str
    kind: str
    middleware: Callable
    expected_signature: str

ALLOWED_MIDDLEWARE_KINDS = {
    "wrap_tool_call",
    "before_model",
    "dynamic_prompt",
}

def _format_registration_error(
    *,
    name: str,
    expected: str,
    actual_kind: str,
    actual_signature: str,
    hint: str,
) -> str:
    return (
        f"Middleware registration failed: {name}\n"
        f"expected: {expected}\n"
        f"actual: kind={actual_kind}, signature={actual_signature}\n"
        f"hint: {hint}"
    )

def _get_signature(callable_obj: Callable) -> inspect.Signature:
    target = inspect.unwrap(callable_obj)
    return inspect.signature(target)


def _validate_signature_by_kind(kind: str, signature: inspect.Signature) -> bool:
    params = list(signature.parameters.values())
    param_names = [param.name for param in params]

    if kind == "wrap_tool_call":
        return len(params) == 2 and param_names == ["request", "handler"]

    if kind == "before_model":
        return len(params) == 2 and param_names == ["state", "runtime"]

    if kind == "dynamic_prompt":
        return len(params) == 1 and param_names == ["request"]

    return False

def _validate_registration(registration: MiddlewareRegistration):
    if registration.kind not in ALLOWED_MIDDLEWARE_KINDS:
        raise MiddlewareRegistrationError(
            _format_registration_error(
                name=registration.name,
                expected=f"kind in {sorted(ALLOWED_MIDDLEWARE_KINDS)}",
                actual_kind=registration.kind,
                actual_signature="N/A",
                hint="use one of wrap_tool_call / before_model / dynamic_prompt",
            )
        )

    if not callable(registration.middleware):
        raise MiddlewareRegistrationError(
            _format_registration_error(
                name=registration.name,
                expected=f"kind={registration.kind}, signature={registration.expected_signature}",
                actual_kind=registration.kind,
                actual_signature="not-callable",
                hint="ensure middleware is a callable decorated by the expected middleware decorator",
            )
        )

    actual_signature = _get_signature(registration.middleware)
    if not _validate_signature_by_kind(registration.kind, actual_signature):
        raise MiddlewareRegistrationError(
            _format_registration_error(
                name=registration.name,
                expected=f"kind={registration.kind}, signature={registration.expected_signature}",
                actual_kind=registration.kind,
                actual_signature=str(actual_signature),
                hint="check decorator and function parameters in agent/tools/middleware.py",
            )
        )

def get_registered_middlewares() -> List[Callable]:
    registrations = [
        MiddlewareRegistration(
            name="monitor_tool",
            kind="wrap_tool_call",
            middleware=monitor_tool,
            expected_signature="(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command",
        ),
        MiddlewareRegistration(
            name="log_before_model",
            kind="before_model",
            middleware=log_before_model,
            expected_signature="(state: AgentState, runtime: Runtime)",
        ),
        MiddlewareRegistration(
            name="report_prompt_switch",
            kind="dynamic_prompt",
            middleware=report_prompt_switch,
            expected_signature="(request: ModelRequest)",
        ),
    ]

    for registration in registrations:
        _validate_registration(registration)

    return [registration.middleware for registration in registrations]

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
    messages = state.get("messages", [])
    logger.info(f"[模型调用监控]即将调用模型，附带{len(messages)}条消息")

    if messages:
        latest = messages[-1]
        logger.debug(f"[模型调用监控] {type(latest).__name__} | {str(latest.content).strip()}")

    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    isReport = request.runtime.context.get("report", False)
    if isReport:
        return load_report_prompt()
    
    return load_system_prompt()


    