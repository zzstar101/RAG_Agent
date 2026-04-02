# RAG_Agent 技术规范

更新时间：2026-04-02

## Agent 中间件注册兼容性规范

### 目标

避免 create_agent 初始化时因 middleware 类型或签名不匹配导致启动失败，并保证失败时可快速定位。

### 适用范围

- agent/react_agent.py
- agent/tools/middleware.py

### 统一注册入口

1. 所有中间件必须通过 get_registered_middlewares() 返回给 create_agent。
2. 禁止在 create_agent(...) 调用处直接手写 middleware 列表。
3. 新增中间件时，必须在统一注册入口增加注册项（名称、种类、目标函数、期望签名说明）。

### 中间件种类约束

仅允许以下种类：

- wrap_tool_call
- before_model
- dynamic_prompt

若出现未知种类，启动阶段必须抛出 MiddlewareRegistrationError。

### 签名校验约束

启动时必须执行中间件校验。校验逻辑要求：

1. 中间件对象必须可调用。
2. 必须校验函数参数结构（使用 inspect.unwrap + inspect.signature）：
   - wrap_tool_call: 参数名必须为 request, handler，且参数数量为 2。
   - before_model: 参数名必须为 state, runtime，且参数数量为 2。
   - dynamic_prompt: 参数名必须为 request，且参数数量为 1。
3. 不允许仅依赖注解字符串全等判断兼容性。

### 错误提示规范

校验失败时抛出 MiddlewareRegistrationError，错误消息应至少包含：

- 中间件名称
- 期望种类与签名说明
- 实际种类与实际签名
- 修复提示

推荐格式：

Middleware registration failed: <name>
expected: kind=<kind>, signature=<expected>
actual: kind=<kind>, signature=<actual>
hint: <how to fix>

### Agent 初始化约束

1. ReactAgent 初始化必须从 get_registered_middlewares() 获取中间件列表。
2. 初始化时应捕获 MiddlewareRegistrationError，记录日志后继续抛出，禁止静默吞错。

### 运行时健壮性补充

before_model 类中间件访问 messages 时必须做空列表保护，避免二次异常掩盖真实注册问题。

### 验收标准

执行以下命令应成功（退出码 0）：

python -m agent.react_agent
