"""Workshop mode: a smaller, faster agent for hands-on sessions on Agent Server.

The full agent (`agent.py`, `graph.py`) boots a LangSmith sandbox per thread and can fan
out to hundreds of subagents; one research question measured about 2,000 runs and up to
11 minutes. That suits a demo, not a room of engineers who each deploy their own copy and
chat with it in Studio. Workshop mode keeps the same sources and analyst leaves and
changes four things:

- **No sandbox.** Files live in graph state (`StateBackend`), so `execute`, figure and
  supplementary staging, uploads and the artifact sweep are gone. Nothing to provision.
- **A small fan-out.** The prompt (`prompts/workshop.py`) caps a question at 8 abstracts
  and 5 trial records, with no full texts.
- **Web search waits for approval.** `web_search` is a direct tool of the root model,
  not a function inside `eval`, so `HumanInTheLoopMiddleware` can pause before it runs.
  A call made from inside `eval` goes through the QuickJS bridge, which no middleware sees.
- **Instructions come from Context Hub at runtime** (`instructions.py`): the repo's
  `production` tag is read on every run, so promoting or rolling back a commit changes the
  next answer with no redeploy, and every trace records the commit it ran.

`langgraph.workshop.json` serves three graphs: `agent` (`graph.py` here), `agent_staging`
(the same agent on the `staging` tag, for trying a change before promoting it to
production), and `setup` (`setup.py`), which a workshop attendee runs once from Studio to
create the judges, the review queue rule, the Context Hub repo and the saved traces in
their own workspace.
"""
