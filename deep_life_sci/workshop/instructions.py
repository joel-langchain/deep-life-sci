"""The agent's answering instructions, read from LangSmith Context Hub on every run.

The code keeps how to use the tools (`prompts/workshop.py`); Context Hub keeps how to
answer: audience, length, citations, sources, scope. That split is what the workshop
exercises. A change to how the agent answers is a commit in Context Hub, it goes live
when the commit is promoted (the `production` tag moves to it), and it is undone by
moving the tag back. The deployment is never rebuilt.

Context Hub has two environment tags, `production` and `staging`. The `agent` graph reads
`production` and the `agent_staging` graph reads `staging`, so a change can be promoted to
staging and tried there before it reaches the graph everyone uses.

`load(tag)` resolves the tag on every call, behind a per-tag cache of `CONTEXT_HUB_TTL`
seconds so that a fan-out of graph builds within one run does not pull once each. Twenty
seconds is short enough that "promote, then ask again" shows the new behaviour on the
next question.

If the hub cannot be read (no repo yet, a network error) the bundled `context/AGENTS.md`
is used and the version says `bundled`, so a run still answers and its trace says why it
used the fallback. `setup.py` seeds the repo from that same file.

Every run records the version it used in its trace metadata (`metadata()`), which is how
an engineer shows that a run before the promote used one commit and a run after it used
another.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The repo and tag are settings so a second copy in one workspace can point elsewhere;
# a workshop never needs to change them.
CONTEXT_HUB_REPO = os.environ.get("CONTEXT_HUB_REPO", "").strip() or "drug-research-assistant"
CONTEXT_HUB_TAG = os.environ.get("CONTEXT_HUB_TAG", "").strip() or "production"
# Context Hub's other environment tag. Fixed: the UI promotes only to these two.
STAGING_TAG = "staging"
CONTEXT_HUB_TTL = float(os.environ.get("CONTEXT_HUB_TTL", "").strip() or "20")

# The one file in the repo. Context Hub agent repos are file trees; the agent reads this.
INSTRUCTIONS_FILE = "AGENTS.md"
BUNDLED = Path(__file__).resolve().parent / "context" / INSTRUCTIONS_FILE


@dataclass(frozen=True)
class Instructions:
    text: str
    repo: str
    tag: str
    # The resolved commit hash, or "bundled" when the hub could not be read.
    commit: str
    error: str | None = None

    def metadata(self) -> dict:
        """Trace metadata: which instructions this run followed."""
        meta = {
            "context_hub_repo": self.repo,
            "context_hub_tag": self.tag,
            "context_hub_commit": self.commit,
        }
        if self.error:
            meta["context_hub_error"] = self.error
        return meta


# Tag -> (when it was read, what was read).
_cache: dict[str, tuple[float, Instructions]] = {}


def bundled_text() -> str:
    return BUNDLED.read_text(encoding="utf-8")


def _pull(tag: str) -> Instructions:
    from langsmith import Client

    try:
        agent = Client().pull_agent(CONTEXT_HUB_REPO, version=tag)
        entry = agent.files.get(INSTRUCTIONS_FILE)
        text = getattr(entry, "content", "") if entry is not None else ""
        if not text.strip():
            raise LookupError(f"{INSTRUCTIONS_FILE} is missing or empty in the repo")
        return Instructions(text, CONTEXT_HUB_REPO, tag, str(agent.commit_hash or "unknown"))
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the bundled copy
        reason = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("Context Hub %s:%s unreadable, using the bundled instructions (%s)",
                       CONTEXT_HUB_REPO, tag, reason)
        return Instructions(bundled_text(), CONTEXT_HUB_REPO, tag, "bundled", reason)


def load(tag: str = CONTEXT_HUB_TAG) -> Instructions:
    """The instructions at `tag` right now, cached for `CONTEXT_HUB_TTL` seconds.

    Blocking HTTP: call it through `asyncio.to_thread` from async code.
    """
    now = time.monotonic()
    cached = _cache.get(tag)
    if cached is not None and now - cached[0] < CONTEXT_HUB_TTL:
        return cached[1]
    found = _pull(tag)
    _cache[tag] = (now, found)
    return found
