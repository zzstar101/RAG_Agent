from typing import Callable, List
import time
from langchain.agents.middleware import before_model, wrap_tool_call, AgentState, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from dataclasses import dataclass
import inspect
from utils.logger_handler import clear_trace_id, ensure_trace_id, logger, set_trace_id
from utils.prompt_loader import load_report_prompt, load_system_prompt

REPORT_CONTEXT_STATE_KEY = "report_state"
REPORT_CONTEXT_EXPIRES_AT_KEY = "report_expires_at"
REPORT_CONTEXT_TTL_SECONDS = 180


def activate_report_context(
    context: dict,
    *,
    now_ts: float | None = None,
    ttl_seconds: int = REPORT_CONTEXT_TTL_SECONDS,
    trace_id: str | None = None,
) -> None:
    timestamp = now_ts if now_ts is not None else time.time()
    context[REPORT_CONTEXT_STATE_KEY] = "active"
    context[REPORT_CONTEXT_EXPIRES_AT_KEY] = timestamp + max(ttl_seconds, 1)
    context["report_entered_at"] = timestamp
    if trace_id:
        context["trace_id"] = trace_id
    logger.info("[报告状态机]进入报告模式，ttl=%ss", max(ttl_seconds, 1))


def deactivate_report_context(
    context: dict,
    *,
    reason: str,
    now_ts: float | None = None,
) -> None:
    timestamp = now_ts if now_ts is not None else time.time()
    context[REPORT_CONTEXT_STATE_KEY] = "inactive"
    context[REPORT_CONTEXT_EXPIRES_AT_KEY] = 0.0
    context["report_last_exit_reason"] = reason
    context["report_exited_at"] = timestamp
    logger.info("[报告状态机]退出报告模式，reason=%s", reason)


def is_report_context_active(context: dict, *, now_ts: float | None = None) -> bool:
    timestamp = now_ts if now_ts is not None else time.time()
    if context.get(REPORT_CONTEXT_STATE_KEY) != "active":
        return False

    expires_at = float(context.get(REPORT_CONTEXT_EXPIRES_AT_KEY, 0.0) or 0.0)
    if timestamp > expires_at:
        deactivate_report_context(context, reason="ttl_expired", now_ts=timestamp)
        return False

    return True


def _record_tool_metrics(context: dict, *, success: bool, duration_ms: float) -> None:
    metrics = context.setdefault(
        "tool_metrics",
        {
            "total": 0,
            "success": 0,
            "error": 0,
            "duration_ms_total": 0.0,
        },
    )
    metrics["total"] += 1
    metrics["duration_ms_total"] += duration_ms
    if success:
        metrics["success"] += 1
    else:
        metrics["error"] += 1

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
    trace_id = request.runtime.context.get("trace_id")
    if trace_id:
        set_trace_id(trace_id)
    else:
        request.runtime.context["trace_id"] = ensure_trace_id()

    started_at = time.perf_counter()
    logger.info(f"[工具调用监控]执行工具：{request.tool_call['name']}")
    logger.info(f"[工具调用监控]传入参数：{request.tool_call['args']}")
    
    try:
        result = handler(request)
        duration_ms = (time.perf_counter() - started_at) * 1000
        _record_tool_metrics(request.runtime.context, success=True, duration_ms=duration_ms)
        logger.info(f"[工具调用监控]工具{request.tool_call['name']}调用成功")
        logger.info(f"[工具调用监控]工具耗时：{duration_ms:.2f}ms")
        
        if request.tool_call['name'] == "fill_context_for_report":
            activate_report_context(request.runtime.context, trace_id=request.runtime.context.get("trace_id"))
            
        return result
    except Exception as e:
        duration_ms = (time.perf_counter() - started_at) * 1000
        _record_tool_metrics(request.runtime.context, success=False, duration_ms=duration_ms)
        logger.error(f"[工具调用监控]工具{request.tool_call['name']}调用失败：{str(e)}")
        logger.error(f"[工具调用监控]工具耗时：{duration_ms:.2f}ms")
        raise e
    finally:
        clear_trace_id()

@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    trace_id = runtime.context.get("trace_id")
    if trace_id:
        set_trace_id(trace_id)
    else:
        runtime.context["trace_id"] = ensure_trace_id()

    previous_started_at = runtime.context.get("model_started_at")
    if isinstance(previous_started_at, (int, float)):
        elapsed_ms = (time.perf_counter() - float(previous_started_at)) * 1000
        logger.info(f"[模型调用监控]上一次模型链路耗时：{elapsed_ms:.2f}ms")

    runtime.context["model_started_at"] = time.perf_counter()

    messages = state.get("messages", [])
    logger.info(f"[模型调用监控]即将调用模型，附带{len(messages)}条消息")

    if messages:
        latest = messages[-1]
        logger.debug(f"[模型调用监控] {type(latest).__name__} | {str(latest.content).strip()}")

    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report_mode = is_report_context_active(request.runtime.context)
    if is_report_mode:
        return load_report_prompt()
    
    return load_system_prompt()


    