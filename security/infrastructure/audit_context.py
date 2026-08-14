import contextvars

# Request-scoped context variables to store request telemetry (client_ip, user_agent, etc.)
# Defaults to empty dict for background execution environments.
audit_context = contextvars.ContextVar("audit_context", default={})
