"""Workshop-mode assembly: the full agent's sources and leaves, minus the sandbox.

Kept beside `agent.py` rather than inside it so the full agent's assembly is unchanged.
Like `agent.build_agent`, nothing here performs I/O: the caller passes the instructions
it already read from Context Hub.

What differs from `agent.build_agent`, and why:

- `StateBackend` instead of a sandbox. With no sandbox there is no `execute`, so the
  tools that stage bytes into one (`fetch_figures`, `fetch_supplementary`, the staging
  trial fetch) and the middleware that reads or publishes sandbox files (uploads,
  artifacts) are left out. `ctgov_fetch` is the plain registry fetch.
- Two leaves, `abstract-analyst` and `trial-analyst`. Full-text, figure and document
  analysts need papers or files a short session never fetches.
- `web_search` is bound to the root model and listed in `interrupt_on`, so a person
  approves each search. Inside `eval` it would run through the QuickJS bridge, which no
  middleware sees. Its return is the search model's digest, not raw pages, so binding it
  directly costs the root at most `web.MAX_ANSWER_CHARS`.
- A tighter interpreter: fewer PTC calls allowed per `eval`, and a shorter timeout.
"""

from __future__ import annotations

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents._models import get_model_provider
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from langchain_quickjs import CodeInterpreterMiddleware

from deep_life_sci.middleware.cadence import UpdateCadence
from deep_life_sci.middleware.model_errors import SurfaceModelErrors
from deep_life_sci.middleware.progress import with_progress
from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.models import (
    CHAT_ROLES,
    report_web_search_problem,
    root_model,
    subagent_model,
    takes_system_midway,
    validate,
)
from deep_life_sci.paths import SKILLS_DIR
from deep_life_sci.prompts.subagents import ABSTRACT_ANALYST, TRIAL_ANALYST
from deep_life_sci.prompts.workshop import build_workshop_prompt
from deep_life_sci.sources.ctgov import ctgov_fetch, ctgov_search
from deep_life_sci.sources.pubmed import fetch_abstracts, pubmed_search
from deep_life_sci.sources.web import web_search

# Tools reached from JavaScript inside `eval` (camelCased there).
PTC_TOOLS = ["pubmed_search", "fetch_abstracts", "ctgov_search", "ctgov_fetch"]

# The approval request Studio shows before a web search runs.
WEB_SEARCH_APPROVAL = {
    "allowed_decisions": ["approve", "reject"],
    "description": (
        "The agent wants to search the open web. Approve to run this search, or reject "
        "to have it answer from PubMed and ClinicalTrials.gov only."
    ),
}


def build_workshop_agent(instructions: str):
    """Assemble the workshop agent with these answering instructions."""
    validate(*CHAT_ROLES)
    report_web_search_problem()
    backend = StateBackend()

    def analyst_leaf(spec: dict) -> dict:
        # Same narrowing as `agent.py`: `tools: []` drops the parent's tools, and the
        # configured FilesystemMiddleware replaces deepagents' default one. The trial
        # analyst's `grep` is for staged files, which never exist here; records arrive
        # inline in its task.
        return {
            **spec,
            "model": subagent_model(),
            "tools": [],
            "middleware": [
                FilesystemMiddleware(backend=backend, tools=["read_file"]),
                SurfaceModelErrors("subagent"),
            ],
        }

    root = root_model()
    leaves = [analyst_leaf(ABSTRACT_ANALYST), analyst_leaf(TRIAL_ANALYST)]

    no_general_purpose = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    built = (get_model_provider(model) for model in (root, *(leaf["model"] for leaf in leaves)))
    for provider in {"openai", "anthropic", *filter(None, built)}:
        register_harness_profile(provider, no_general_purpose)

    skills_route = "/skills/"
    root_backend = CompositeBackend(
        default=backend,
        routes={skills_route: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
    )

    source_tools = with_error_capture(with_progress([
        pubmed_search,
        fetch_abstracts,
        ctgov_search,
        ctgov_fetch,
    ]))
    search_tool = with_error_capture([web_search])

    agent = create_deep_agent(
        model=root,
        tools=[*source_tools, *search_tool],
        system_prompt=build_workshop_prompt(instructions),
        subagents=leaves,
        backend=root_backend,
        skills=[(skills_route, "Research")],
        permissions=[
            FilesystemPermission(operations=["write"], paths=[f"{skills_route}**"], mode="deny")
        ],
        interrupt_on={"web_search": WEB_SEARCH_APPROVAL},
        middleware=[
            SurfaceModelErrors("root"),
            UpdateCadence(system_turns=takes_system_midway(root)),
            CodeInterpreterMiddleware(
                ptc=PTC_TOOLS,
                timeout=300.0,
                max_result_chars=20_000,
                max_ptc_calls=64,
            ),
        ],
    )
    return agent.with_config(recursion_limit=100)
