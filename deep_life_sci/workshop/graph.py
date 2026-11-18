"""Server entry points for workshop mode: the `agent` and `agent_staging` graphs in
`langgraph.workshop.json`.

Called per request like `deep_life_sci.graph.make_graph`. A run reads the Context Hub
instructions at its tag (`instructions.load`, cached briefly) and records the tag and
commit in its trace metadata, so promoting a commit changes the next run and the trace
shows which version answered. `agent` reads `production` and `agent_staging` reads
`staging`; the agent is otherwise the same. A read (Studio drawing the graph, a thread's
history) gets the bundled instructions instead, since its prompt is never sent anywhere.

No sandbox is acquired, so there is nothing to clean up and nothing to keep warm.
"""

from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime

from deep_life_sci import models
from deep_life_sci.middleware.perf import install_logging
from deep_life_sci.workshop import instructions
from deep_life_sci.workshop.agent import build_workshop_agent

install_logging()

# A models-file mistake stops the server at start, with its message.
models.refresh()
models.validate(*models.CHAT_ROLES)
models.report_web_search_problem()


def _build(text: str):
    try:
        return build_workshop_agent(text)
    except SystemExit as exc:
        # Config checks raise SystemExit; inside a request that must be an ordinary error.
        raise RuntimeError(str(exc)) from None


async def _make(runtime: ServerRuntime, tag: str):
    try:
        await asyncio.to_thread(models.refresh)
        models.validate(*models.CHAT_ROLES)
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from None

    if runtime.execution_runtime is None:
        return _build(instructions.bundled_text())

    found = await asyncio.to_thread(instructions.load, tag)
    return _build(found.text).with_config(metadata=found.metadata())


async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    """The workshop agent with the instructions at the `production` tag.

    `runtime` must stay annotated as `ServerRuntime`: the server passes it only then, and
    `execution_runtime` is what tells a run from a read.
    """
    return await _make(runtime, instructions.CONTEXT_HUB_TAG)


async def make_staging_graph(config: RunnableConfig, runtime: ServerRuntime):
    """The same agent with the instructions at the `staging` tag."""
    return await _make(runtime, instructions.STAGING_TAG)
