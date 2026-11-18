"""Workshop mode: assembly, the approval before web search, Context Hub instructions, setup.

All offline. The compiled-agent tests run the real middleware stack with a scripted
model, so they catch the wiring that matters in the room: `web_search` stops for approval
and runs only once approved, and `eval` still reaches the sources.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_life_sci.workshop import agent as workshop_agent
from deep_life_sci.workshop import instructions, setup, traces

RUN = SimpleNamespace(execution_runtime=SimpleNamespace())
READ = SimpleNamespace(execution_runtime=None)


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def fake_search(monkeypatch):
    calls: list[str] = []

    @tool
    async def web_search(query: str) -> dict:
        """Search the web."""
        calls.append(query)
        return {"answer": "Label updated in 2025.", "sources": ["https://example.gov/label"]}

    monkeypatch.setattr(workshop_agent, "web_search", web_search)
    return calls


def _compile(monkeypatch, responses):
    model = ScriptedModel(responses=responses)
    monkeypatch.setattr(workshop_agent, "root_model", lambda: model)
    monkeypatch.setattr(workshop_agent, "subagent_model", lambda: model)
    monkeypatch.setattr(workshop_agent, "register_harness_profile", Mock())
    compiled = workshop_agent.build_workshop_agent("Cite everything.")
    compiled.checkpointer = InMemorySaver()
    return compiled


def _search_call():
    return AIMessage("", tool_calls=[{
        "name": "web_search", "id": "ws1", "args": {"query": "FDA label change 2025?"}}])


def test_assembly_has_no_sandbox_tools_and_puts_instructions_in_the_prompt(monkeypatch):
    model = ScriptedModel(responses=[AIMessage("unused")])
    monkeypatch.setattr(workshop_agent, "root_model", lambda: model)
    monkeypatch.setattr(workshop_agent, "subagent_model", lambda: model)
    monkeypatch.setattr(workshop_agent, "register_harness_profile", Mock())
    create = Mock()
    monkeypatch.setattr(workshop_agent, "create_deep_agent", create)
    workshop_agent.build_workshop_agent("MARKER: cite PMIDs.")
    kwargs = create.call_args.kwargs
    names = {t.name for t in kwargs["tools"]}
    assert names == {"pubmed_search", "fetch_abstracts", "ctgov_search", "ctgov_fetch",
                     "web_search"}
    assert kwargs["system_prompt"].rstrip().endswith("MARKER: cite PMIDs.")
    assert "web_search" in kwargs["interrupt_on"]
    assert set(kwargs["interrupt_on"]["web_search"]["allowed_decisions"]) == {"approve",
                                                                               "reject"}
    assert [leaf["name"] for leaf in kwargs["subagents"]] == ["abstract-analyst",
                                                               "trial-analyst"]
    interpreter = next(m for m in kwargs["middleware"] if type(m).__name__ ==
                       "CodeInterpreterMiddleware")
    assert "web_search" not in workshop_agent.PTC_TOOLS
    assert interpreter is not None


async def test_web_search_waits_for_approval_then_runs(monkeypatch, fake_search):
    compiled = _compile(monkeypatch, [_search_call(), AIMessage("Answer with a URL.")])
    config = {"configurable": {"thread_id": "t1"}}
    first = await compiled.ainvoke({"messages": [HumanMessage("Label change?")]}, config)
    assert "__interrupt__" in first
    request = first["__interrupt__"][0].value
    assert request["action_requests"][0]["name"] == "web_search"
    assert fake_search == []  # nothing ran before the approval

    done = await compiled.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert fake_search == ["FDA label change 2025?"]
    assert done["messages"][-1].text == "Answer with a URL."


async def test_rejected_web_search_never_runs(monkeypatch, fake_search):
    compiled = _compile(monkeypatch, [_search_call(), AIMessage("Answered without the web.")])
    config = {"configurable": {"thread_id": "t2"}}
    await compiled.ainvoke({"messages": [HumanMessage("Label change?")]}, config)
    done = await compiled.ainvoke(
        Command(resume={"decisions": [{"type": "reject", "message": "Not approved."}]}), config)
    assert fake_search == []
    rejected = [m for m in done["messages"] if isinstance(m, ToolMessage)]
    assert rejected and rejected[-1].status == "error"
    assert done["messages"][-1].text == "Answered without the web."


async def test_eval_still_reaches_pubmed(monkeypatch, mock_ncbi, fake_search):
    import httpx

    mock_ncbi(lambda req: httpx.Response(
        200, json={"esearchresult": {"count": "7", "idlist": [], "querytranslation": "x"}}))
    compiled = _compile(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "eval", "id": "e1", "args": {
            "code": 'const r = await tools.pubmedSearch({term: "x", retmax: 0}); '
                    'console.log("COUNT=" + r.count);'}}]),
        AIMessage("Found 7."),
    ])
    done = await compiled.ainvoke({"messages": [HumanMessage("Count")]},
                                  {"configurable": {"thread_id": "t3"}})
    output = next(m for m in done["messages"] if isinstance(m, ToolMessage))
    assert "COUNT=7" in output.text
    assert done["messages"][-1].text == "Found 7."


def test_unreadable_hub_falls_back_to_the_bundled_text(monkeypatch):
    import langsmith

    client = Mock()
    client.pull_agent.side_effect = RuntimeError("no such repo")
    monkeypatch.setattr(langsmith, "Client", lambda: client)
    monkeypatch.setattr(instructions, "_cache", {})
    found = instructions.load()
    assert found.commit == "bundled"
    assert found.text == instructions.bundled_text()
    meta = found.metadata()
    assert meta["context_hub_commit"] == "bundled"
    assert "no such repo" in meta["context_hub_error"]


def test_hub_text_and_commit_are_used_and_cached(monkeypatch):
    import langsmith

    client = Mock()
    client.pull_agent.return_value = SimpleNamespace(
        files={"AGENTS.md": SimpleNamespace(content="From the hub.")}, commit_hash="abc12345")
    monkeypatch.setattr(langsmith, "Client", lambda: client)
    monkeypatch.setattr(instructions, "_cache", {})
    first = instructions.load()
    second = instructions.load()
    assert (first.text, first.commit) == ("From the hub.", "abc12345")
    assert second is first
    client.pull_agent.assert_called_once_with(instructions.CONTEXT_HUB_REPO,
                                              version=instructions.CONTEXT_HUB_TAG)


def test_staging_is_read_and_cached_apart_from_production(monkeypatch):
    """Promoting to staging must not change what the production graph reads, or the
    other way round, even inside the cache window."""
    import langsmith

    texts = {"production": "Live text.", "staging": "Trial text."}
    client = Mock()
    client.pull_agent.side_effect = lambda repo, version: SimpleNamespace(
        files={"AGENTS.md": SimpleNamespace(content=texts[version])},
        commit_hash=f"{version[:4]}0000")
    monkeypatch.setattr(langsmith, "Client", lambda: client)
    monkeypatch.setattr(instructions, "_cache", {})
    live = instructions.load()
    trial = instructions.load(instructions.STAGING_TAG)
    assert (live.text, live.tag) == ("Live text.", "production")
    assert (trial.text, trial.tag) == ("Trial text.", "staging")
    assert trial.metadata()["context_hub_tag"] == "staging"
    assert instructions.load() is live and instructions.load("staging") is trial
    assert client.pull_agent.call_count == 2


async def test_graph_reads_use_bundled_text_and_runs_record_the_version(monkeypatch):
    from deep_life_sci.workshop import graph

    built: list[str] = []
    compiled = Mock()
    monkeypatch.setattr(graph, "build_workshop_agent",
                        lambda text: built.append(text) or compiled)
    load = Mock(return_value=instructions.Instructions("Hub text", "repo", "production",
                                                       "abc12345"))
    monkeypatch.setattr(instructions, "load", load)

    await graph.make_graph({}, READ)
    load.assert_not_called()
    assert built[-1] == instructions.bundled_text()

    result = await graph.make_graph({"configurable": {"thread_id": "t"}}, RUN)
    assert built[-1] == "Hub text"
    compiled.with_config.assert_called_once_with(metadata={
        "context_hub_repo": "repo", "context_hub_tag": "production",
        "context_hub_commit": "abc12345"})
    assert result is compiled.with_config.return_value
    load.assert_called_once_with(instructions.CONTEXT_HUB_TAG)


async def test_staging_graph_reads_the_staging_tag(monkeypatch):
    from deep_life_sci.workshop import graph

    compiled = Mock()
    monkeypatch.setattr(graph, "build_workshop_agent", lambda text: compiled)
    load = Mock(return_value=instructions.Instructions("Trial", "repo", "staging", "beef0000"))
    monkeypatch.setattr(instructions, "load", load)
    await graph.make_staging_graph({"configurable": {"thread_id": "t"}}, RUN)
    load.assert_called_once_with("staging")
    assert compiled.with_config.call_args.kwargs["metadata"]["context_hub_tag"] == "staging"


def test_workshop_config_serves_agent_staging_and_setup():
    import json
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "langgraph.workshop.json")
                        .read_text())
    assert config["graphs"] == {
        "agent": "./deep_life_sci/workshop/graph.py:make_graph",
        "agent_staging": "./deep_life_sci/workshop/graph.py:make_staging_graph",
        "setup": "./deep_life_sci/workshop/setup.py:graph",
    }


def test_bundled_instructions_carry_the_planted_citation_problem():
    """The saved traces and the guide's fix depend on this exact baseline."""
    text = instructions.bundled_text()
    assert "Leave out reference lists, PMIDs, and NCT numbers." in text
    assert "northpoint.example" in text


def test_context_hub_instructions_outrank_the_skills_on_citations():
    """The clinical-trials skill says to cite NCT numbers. If it won, the baseline would
    cite on trial questions and the planted problem would come and go."""
    from deep_life_sci.prompts.workshop import build_workshop_prompt

    prompt = build_workshop_prompt("BASELINE")
    assert ("citations included, comes from the\n"
            "Instructions section below, not from the skills") in prompt
    assert prompt.index("not from the skills") < prompt.index("BASELINE")


class _Client:
    """Records LangSmith API calls; answers the handful setup reads."""

    def __init__(self, rules=(), queues=(), tags_exist=False):
        self.calls: list[tuple[str, str, dict]] = []
        self._rules = list(rules)
        self._queues = list(queues)
        self._tags_exist = tags_exist

    def request_with_retries(self, method, path, request_kwargs=None):
        from langsmith.utils import LangSmithError

        self.calls.append((method, path, request_kwargs or {}))
        if (method, path) == ("POST", "/feedback-configs") and self._tags_exist:
            raise LangSmithError("400 a different config exists")
        body = None
        if (method, path) == ("GET", "/runs/rules"):
            body = self._rules
        elif (method, path) == ("GET", "/annotation-queues"):
            body = self._queues
        elif (method, path) == ("POST", "/annotation-queues"):
            body = {"id": "q1", "name": setup.QUEUE}
        return SimpleNamespace(json=lambda: body)


def test_rules_score_and_route_only_agent_root_runs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = _Client(rules=[
        {"id": "old", "session_id": "p1", "display_name": "cites_sources"},
        {"id": "theirs", "session_id": "p1", "display_name": "someone else's rule"},
    ])
    setup.setup_rules(client, "p1", log=lambda _: None)
    deleted = [path for method, path, _ in client.calls if method == "DELETE"]
    assert deleted == ["/runs/rules/old"]
    posted = [kw["json"] for method, path, kw in client.calls
              if (method, path) == ("POST", "/runs/rules")]
    judges = [p for p in posted if "evaluators" in p]
    queue_rules = [p for p in posted if "add_to_annotation_queue_id" in p]
    assert [j["display_name"] for j in judges] == ["cites_sources", "no_individual_advice",
                                                    "states_gaps"]
    assert all(p["filter"].startswith('and(eq(is_root, true), eq(name, "agent")')
               for p in posted)
    assert {('eq(feedback_key, "user_rating")' in q["filter"]) for q in queue_rules} == {
        True, False}
    assert all(q["add_to_annotation_queue_id"] == "q1" for q in queue_rules)
    assert "placeholder" not in str(judges)


def test_a_paused_run_passes_every_judge():
    # A run that stops for web search approval is its own trace with no answer in it.
    for judge in setup.JUDGES:
        assert "paused" in judge["prompt"] and "score 1" in judge["prompt"]
    # The score field says it too: the judge follows the schema over the prompt.
    assert "paused for approval, is 1" in setup.SCORE_FIELD


def test_a_study_cited_once_can_be_mentioned_again():
    # On 18 Nov a fixed answer cited ORION-9 in a table, then failed for a later sentence
    # about ORION-9 with no identifier beside it.
    prompt = next(j["prompt"] for j in setup.JUDGES if j["key"] == "cites_sources")
    assert "identified once can be mentioned again" in prompt
    assert "same row" not in prompt and "near it" not in prompt


def test_rating_tags_exist_and_the_queue_asks_for_a_score(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = _Client()
    setup.setup_rules(client, "p1", log=lambda _: None)
    tags = [kw["json"] for method, path, kw in client.calls
            if (method, path) == ("POST", "/feedback-configs")]
    assert [t["feedback_key"] for t in tags] == ["user_rating", "correct"]
    assert all(t["feedback_config"] == {"type": "continuous", "min": 0, "max": 1}
               for t in tags)
    created = next(kw["json"] for method, path, kw in client.calls
                   if (method, path) == ("POST", "/annotation-queues"))
    assert [r["feedback_key"] for r in created["rubric_items"]] == ["correct"]


def test_an_existing_queue_gets_the_score_and_existing_tags_are_kept(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    client = _Client(queues=[{"id": "q9", "name": setup.QUEUE}], tags_exist=True)
    notes: list[str] = []
    setup.setup_rules(client, "p1", log=notes.append)
    patched = [kw["json"] for method, path, kw in client.calls
               if (method, path) == ("PATCH", "/annotation-queues/q9")]
    assert patched == [{"rubric_items": setup.QUEUE_RUBRIC}]
    assert not any(path == "/annotation-queues" and method == "POST"
                   for method, path, _ in client.calls)
    assert sum("kept as it is" in n for n in notes) == 2


def test_existing_instructions_are_kept_unless_reset():
    client = _Client()
    client.pull_agent = Mock(return_value=SimpleNamespace(commit_hash="c0ffee00",
                                                          commit_id="id1"))
    client.push_agent = Mock()
    result = setup.setup_context_hub(client, reset=False, log=lambda _: None)
    assert result["seeded"] is False
    client.push_agent.assert_not_called()

    result = setup.setup_context_hub(client, reset=True, log=lambda _: None)
    assert result["seeded"] is True
    client.push_agent.assert_called_once()
    tagged = [kw["json"] for method, path, kw in client.calls if path.endswith("/tags")]
    assert tagged == [{"tag_name": "production", "commit_id": "id1"},
                      {"tag_name": "staging", "commit_id": "id1"}]


def test_a_kept_repo_without_staging_gets_it_at_the_production_commit():
    from langsmith.utils import LangSmithError

    def pull(repo, version=None):
        if version == "staging":
            raise LangSmithError("no such tag")
        return SimpleNamespace(commit_hash="c0ffee00", commit_id="prod-id")

    client = _Client()
    client.pull_agent = Mock(side_effect=pull)
    client.push_agent = Mock()
    result = setup.setup_context_hub(client, reset=False, log=lambda _: None)
    assert result["seeded"] is False
    client.push_agent.assert_not_called()
    tagged = [kw["json"] for method, path, kw in client.calls if path.endswith("/tags")]
    assert tagged == [{"tag_name": "staging", "commit_id": "prod-id"}]


def test_ratings_follow_whether_the_answer_can_be_traced():
    import random

    cited = {"outputs": {"messages": [{"type": "ai", "content": "See PMID 123."}]}}
    uncited = {"outputs": {"messages": [{"type": "ai", "content": "Three trials found."}]}}
    interrupted = {"outputs": {"messages": [{"type": "ai", "content": "",
                                             "tool_calls": [{"name": "web_search"}]}]}}
    rng = random.Random(0)
    assert traces.rate(cited, rng)[0] == 1
    assert {traces.rate(uncited, random.Random(s)) for s in range(40)} - {None} <= {
        (0, note) for note in traces._UNCITED_NOTES}
    assert traces.rate(interrupted, rng) is None


def test_dotted_orders_nest_children_under_parents():
    from datetime import datetime

    t = datetime(2026, 11, 20, 13, 30)
    runs = [{"id": "r", "parent_run_id": None, "start_time": t},
            {"id": "c", "parent_run_id": "r", "start_time": t},
            {"id": "g", "parent_run_id": "c", "start_time": t}]
    orders = traces._dotted_orders(runs)
    assert orders["g"].startswith(orders["c"] + ".")
    assert orders["c"].startswith(orders["r"] + ".")


def test_setup_without_saved_traces_skips_the_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(traces, "SAVED", tmp_path / "missing.json.gz")
    client = Mock()
    logged: list[str] = []
    result = setup.setup_traces(client, "p", "p1", log=logged.append)
    assert result["skipped"] is True
    client.list_runs.assert_not_called()
    assert "own chats" in logged[0]


def test_setup_finds_the_hosted_project(monkeypatch):
    from langsmith.utils import get_tracer_project

    get_tracer_project.cache_clear()
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.setenv("HOSTED_LANGSERVE_PROJECT_NAME", "drug-research-production")
    try:
        assert setup._current_project() == "drug-research-production"
    finally:
        get_tracer_project.cache_clear()


def test_setup_refuses_the_default_project(monkeypatch):
    from langsmith.utils import get_tracer_project

    get_tracer_project.cache_clear()
    for name in ("HOSTED_LANGSERVE_PROJECT_NAME", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT",
                 "LANGSMITH_SESSION", "LANGCHAIN_SESSION"):
        monkeypatch.delenv(name, raising=False)
    try:
        with pytest.raises(setup.SetupError):
            setup._current_project()
    finally:
        get_tracer_project.cache_clear()


async def test_tools_called_from_eval_are_runs_in_the_same_trace(monkeypatch, mock_ncbi,
                                                                 fake_search):
    """Criterion 1: one trace holds the agent, its subagents, and every tool call.

    Tools reached from `eval` run through the QuickJS bridge with no callbacks of their
    own. They join the trace through LangSmith's tracing context, which langchain-core
    sets only for a `LangChainTracer`, so this traces with one (posting to a mock client).
    """
    import httpx
    from langchain_core.tracers.langchain import LangChainTracer

    mock_ncbi(lambda req: httpx.Response(
        200, json={"esearchresult": {"count": "7", "idlist": [], "querytranslation": "x"}}))
    compiled = _compile(monkeypatch, [
        AIMessage("", tool_calls=[{"name": "eval", "id": "e1", "args": {
            "code": 'const r = await tools.pubmedSearch({term: "x", retmax: 0}); '
                    'console.log(r.count);'}}]),
        AIMessage("Found 7."),
    ])
    client = Mock()
    tracer = LangChainTracer(client=client, project_name="workshop-test")
    await compiled.ainvoke({"messages": [HumanMessage("Count")]},
                           {"configurable": {"thread_id": "t4"}, "callbacks": [tracer]})
    tracer.wait_for_futures()
    posted = [call.kwargs for call in client.create_run.call_args_list]
    by_name = {run["name"]: run for run in posted}
    assert {"eval", "pubmed_search"} <= set(by_name)
    assert len({str(run["trace_id"]) for run in posted}) == 1
    assert by_name["pubmed_search"]["parent_run_id"] == by_name["eval"]["id"]


def test_capture_keeps_answered_agent_traces_only(tmp_path):
    import json
    from datetime import datetime

    def root(name, trace_id, content, tool_calls=None):
        message = {"type": "ai", "content": content, "tool_calls": tool_calls or []}
        return SimpleNamespace(name=name, trace_id=trace_id,
                               outputs={"messages": [message]})

    roots = [
        root("agent", "t1", "Three trials found."),
        root("agent", "t2", "", [{"name": "web_search"}]),  # paused for approval
        root("agent_staging", "t3", "Cited PMID 1."),
        root("setup", "t4", "Done."),
        root("agent", "t5", "Two papers found."),
    ]

    class Client:
        def __init__(self):
            self.root_kwargs = None

        def read_project(self, project_name):
            return SimpleNamespace(id="p1")

        def list_runs(self, project_id, is_root=None, trace_id=None, start_time=None):
            if is_root:
                self.root_kwargs = {"start_time": start_time}
                return iter(roots)
            return [SimpleNamespace(model_dump_json=lambda t=trace_id: json.dumps(
                {"id": f"{t}-run", "trace_id": t, "parent_run_id": None}))]

        def list_feedback(self, run_ids):
            return []

    client = Client()
    since = datetime(2026, 11, 18, 18, 0)
    out = tmp_path / "saved.json.gz"
    assert traces.capture(client, "p", out=out, limit_traces=40, since=since) == 2
    assert client.root_kwargs == {"start_time": since}
    assert {run["trace_id"] for run in traces.load_saved(out)} == {"t1", "t5"}


def test_prune_drops_middleware_spans_and_reattaches_their_children():
    def run(id_, name, parent):
        return {"id": id_, "trace_id": "t", "name": name, "parent_run_id": parent}

    runs = [
        run("r", "agent", None),
        run("m", "model", "r"),
        run("h1", "FilesystemMiddleware.awrap_model_call", "m"),
        run("h2", "SurfaceModelErrors.awrap_model_call", "h1"),
        run("llm", "ChatOpenAI", "h2"),
        run("b", "SkillsMiddleware.before_model", "r"),
        run("t", "tools", "r"),
        run("h3", "HumanInTheLoopMiddleware.awrap_tool_call", "t"),
        run("w", "web_search", "h3"),
        run("a", "CodeInterpreterMiddleware.after_agent", "r"),
    ]
    kept = {r["id"]: r["parent_run_id"] for r in traces.prune(runs)}
    assert kept == {"r": None, "m": "r", "llm": "m", "t": "r", "w": "t"}
    # A root keeps its place even if its name looked like a hook.
    assert traces.prune([run("x", "Odd.before_agent", None)])[0]["id"] == "x"


def test_prune_drops_the_source_workspaces_ids_from_metadata():
    metadata = {"LANGSMITH_TENANT_ID": "ws", "user_id": "u", "langgraph_api_url": "https://x",
                "context_hub_commit": "c", "context_hub_tag": "production", "graph_id": "agent"}
    run = {"id": "r", "trace_id": "r", "name": "agent", "parent_run_id": None,
           "extra": {"metadata": metadata, "runtime": {"sdk": "py"}}}
    (kept,) = traces.prune([run])
    kept_metadata = {
        "context_hub_commit": "c", "context_hub_tag": "production", "graph_id": "agent"}
    assert kept["extra"] == {"metadata": kept_metadata, "runtime": {"sdk": "py"}}
    assert run["extra"]["metadata"]["user_id"] == "u"  # the input is not changed


def test_replayed_traces_name_the_workspaces_own_commit():
    extra = {"metadata": {"context_hub_commit": "reference", "context_hub_tag": "production"}}
    assert traces._with_commit(extra, "attendee") == {
        "metadata": {"context_hub_commit": "attendee", "context_hub_tag": "production"}}
    assert traces._with_commit(extra, None) is extra
    assert traces._with_commit({"metadata": {"other": 1}}, "attendee") == {
        "metadata": {"other": 1}}
    assert extra["metadata"]["context_hub_commit"] == "reference"


@pytest.mark.parametrize("seeded, expected", [(True, "abc123"), (False, None)])
def test_setup_passes_the_seeded_commit_to_the_replay(monkeypatch, seeded, expected):
    import langsmith

    passed = {}
    monkeypatch.setattr(langsmith, "Client", lambda: Mock())
    monkeypatch.setattr(setup, "_project_id", lambda client, project: "p1")
    monkeypatch.setattr(setup, "setup_context_hub", lambda client, reset, log: {
        "repo": "r", "tag": "production", "commit": "abc123", "seeded": seeded})
    monkeypatch.setattr(setup, "setup_rules", lambda client, project_id, log: {})

    def fake_traces(client, project, project_id, *, commit=None, log=print):
        passed["commit"] = commit
        return {}

    monkeypatch.setattr(setup, "setup_traces", fake_traces)
    setup.run_setup("p", log=lambda _: None)
    assert passed["commit"] == expected
