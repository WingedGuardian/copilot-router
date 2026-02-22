# copilot-router

**LLM routing with circuit breaker, self-escalation, and multi-provider failover chains.**

## Features

- **Circuit breaker pattern**: Automatically stops sending requests to failing providers, with configurable recovery timeout
- **Self-escalation**: When the LLM recognizes it can't handle a task (outputs `[ESCALATE]`), automatically retries with a more capable model
- **Multi-provider failover**: Chain of providers tried in order (e.g., local LM Studio → cloud cheap tier → cloud powerful tier)
- **Provider tiers**: Organize providers by capability and cost (primary, secondary, emergency)
- **Health tracking**: Per-provider success/failure tracking with automatic recovery detection
- **Pluggable providers**: Implement the `LLMProvider` ABC to add any LLM backend

## Installation

```bash
pip install copilot-router
```

## Quick Start

```python
import asyncio
from copilot_router import LLMProvider, LLMResponse, FailoverChain, ProviderTier

# Implement the LLMProvider interface for your backend
class MyCloudProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, **kwargs):
        # Your API call here
        return LLMResponse(content="Hello from the cloud!")

class MyLocalProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, **kwargs):
        return LLMResponse(content="Hello from local!")

async def main():
    # Create a failover chain: try local first, then cloud
    chain = FailoverChain(tiers=[
        ProviderTier(name="local", provider=MyLocalProvider(), models=["llama-3"]),
        ProviderTier(name="cloud", provider=MyCloudProvider(), models=["gpt-4o"]),
    ])

    # The chain tries each tier in order until one succeeds
    response = await chain.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        model="llama-3",
    )
    print(response.content)

asyncio.run(main())
```

## Circuit Breaker

Each provider in the chain has an independent circuit breaker:

```python
from copilot_router import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=3,     # Open after 3 consecutive failures
    recovery_timeout=60,     # Try again after 60 seconds
)

# The circuit breaker tracks provider health automatically
# When a provider fails 3 times, it's "tripped" and skipped
# After recovery_timeout, it enters "half-open" state and tries one request
# If that succeeds, the circuit closes again
```

## Self-Escalation

The router can detect when an LLM admits it can't handle a task:

```
User: "Prove the Riemann hypothesis"
Local LLM: "[ESCALATE] This requires deeper mathematical reasoning..."
Router: *automatically retries with escalation_model (e.g., Claude Opus)*
```

Configure by setting an `escalation_model` that's more capable than the default.

## Architecture

```
┌─────────────────────────────────────────┐
│            FailoverChain                 │
│                                          │
│  Tier 1: Local (LM Studio)              │
│    ├── CircuitBreaker                    │
│    └── Provider.chat()                   │
│         ↓ fails                          │
│  Tier 2: Cloud Cheap (Haiku/Flash)       │
│    ├── CircuitBreaker                    │
│    └── Provider.chat()                   │
│         ↓ fails                          │
│  Tier 3: Cloud Powerful (Sonnet/GPT-4o)  │
│    ├── CircuitBreaker                    │
│    └── Provider.chat()                   │
│         ↓ fails                          │
│  Emergency: Last-resort model            │
└─────────────────────────────────────────┘
```

## License

MIT
