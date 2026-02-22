"""Basic structure tests for copilot_router."""

def test_imports():
    from copilot_router import LLMProvider, LLMResponse, RouteDecision, FailoverChain, CircuitBreaker

def test_route_decision():
    from copilot_router import RouteDecision
    rd = RouteDecision(target="cloud", reason="user request", model="gpt-4o")
    assert rd.target == "cloud"

def test_llm_response():
    from copilot_router import LLMResponse
    r = LLMResponse(content="hello")
    assert not r.has_tool_calls
    assert r.content == "hello"

def test_circuit_breaker():
    from copilot_router import CircuitBreaker
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    assert cb.is_closed
