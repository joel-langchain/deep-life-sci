"""Saved traces: captured once from real runs, replayed into a workshop workspace.

Engine needs traces to find an issue, and a room of new deployments has almost none in
the first half hour. So `setup.py` replays a fixed set captured from this agent on the
bundled instructions (`context/AGENTS.md`), and every attendee's Engine reads the same
failures. Same approach as the Engine workshop's starter traces.

    capture:  python -m deep_life_sci.workshop.traces capture --project <source project>
    replay:   setup.py calls `upload()`, or
              python -m deep_life_sci.workshop.traces upload --project <target project>

Replay gives every run a new id, keeps each trace's internal timing, and scatters the
traces over the last `MAX_BACKDATE_DAYS` (ingest rejects start times more than 24 hours
old). Root runs carry the `user_rating` a scientist left in the app, captured with the
trace, so the review queue rule has something to route before anyone rates a live answer.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

SAVED = Path(__file__).resolve().parent / "saved_traces.json.gz"

MAX_BACKDATE_DAYS = 0.9
RATING_KEY = "user_rating"
# The agent graph's root run name on Agent Server, the runs the judges score.
ROOT_NAME = "agent"
# Trace metadata naming the Context Hub version that wrote an answer (`graph.py`).
COMMIT_KEY = "context_hub_commit"

# Identifiers a scientist would look for to check an answer.
_CITATION = re.compile(r"\b(PMID|NCT\d{8}|pubmed\.ncbi|clinicaltrials\.gov/study|doi\.org)", re.I)

_UNCITED_NOTES = (
    "Where are the references? I can't check any of this.",
    "No PMIDs or trial numbers, so I can't use this in a document.",
    "Which trials is this from? Can't verify without the NCT numbers.",
    None,
    None,
)
_GOOD_NOTES = ("Useful, thanks", None, None, "Good summary")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def final_answer(root: dict) -> str:
    """The last assistant message's text from a root run's outputs, or ""."""
    messages = ((root.get("outputs") or {}).get("messages")) or []
    for message in reversed(messages):
        kind = message.get("type") or message.get("role") or ""
        if kind not in ("ai", "assistant", "AIMessage", "AIMessageChunk"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        if isinstance(content, str) and content.strip() and not message.get("tool_calls"):
            return content
    return ""


def rate(root: dict, rng: random.Random) -> tuple[int, str | None] | None:
    """The rating a scientist leaves on this answer, or None for a run with no answer.

    An interrupted run (stopped for web search approval) has no answer to rate.
    """
    answer = final_answer(root)
    if not answer:
        return None
    if _CITATION.search(answer):
        return 1, rng.choice(_GOOD_NOTES)
    # Not every scientist rates a bad answer, and not every one explains.
    if rng.random() < 0.25:
        return None
    return 0, rng.choice(_UNCITED_NOTES)


# --- capture -----------------------------------------------------------------------------

def capture(client, project: str, out: Path = SAVED, limit_traces: int = 40,
            since: datetime | None = None) -> int:
    """Save every run of the last `limit_traces` answered `agent` traces in `project`.

    Only root runs named `agent` with a final answer are kept: `setup` and `agent_staging`
    runs are not what the judges score, and a run paused for web search approval has no
    answer (its resume is a trace of its own and is kept). `since` drops anything older,
    so test runs made before the capture session stay out.
    """
    from langsmith.utils import LangSmithNotFoundError

    try:
        project_id = client.read_project(project_name=project).id
    except LangSmithNotFoundError:
        sys.exit(f"project {project!r} not found")
    roots = []
    for root in client.list_runs(project_id=project_id, is_root=True, start_time=since):
        if root.name != ROOT_NAME or not final_answer({"outputs": root.outputs}):
            continue
        roots.append(root)
        if len(roots) == limit_traces:
            break
    runs = []
    for root in roots:
        for run in client.list_runs(project_id=project_id, trace_id=root.trace_id):
            dump = getattr(run, "model_dump_json", None) or run.json
            runs.append(json.loads(dump()))
    feedback = defaultdict(list)
    for item in client.list_feedback(run_ids=[r["id"] for r in runs if not r.get("parent_run_id")]):
        feedback[str(item.run_id)].append(
            {"key": item.key, "score": item.score, "value": item.value, "comment": item.comment}
        )
    for run in runs:
        run["feedback"] = feedback.get(str(run["id"]), [])
        # Session ids, reference examples and the source workspace's ids mean nothing in
        # the target workspace.
        for field in ("session_id", "reference_example_id", "app_path", "child_run_ids",
                      "direct_child_run_ids", "parent_run_ids", "feedback_stats", "share_token",
                      "manifest_id", "events", "in_dataset", "inputs_s3_urls", "outputs_s3_urls",
                      "s3_urls", "attachments"):
            run.pop(field, None)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(runs, fh)
    print(f"saved {len(roots)} traces, {len(runs)} runs, to {out}")
    return len(runs)


# --- replay ------------------------------------------------------------------------------

def _dotted_orders(trace_runs: list[dict]) -> dict[str, str]:
    """Each run's dotted order, built by walking its parent chain."""
    by_id = {r["id"]: r for r in trace_runs}
    orders: dict[str, str] = {}

    def dotted(run: dict) -> str:
        if run["id"] not in orders:
            segment = run["start_time"].strftime("%Y%m%dT%H%M%S%f") + "Z" + run["id"]
            parent = by_id.get(run["parent_run_id"])
            orders[run["id"]] = segment if parent is None else f"{dotted(parent)}.{segment}"
        return orders[run["id"]]

    for run in trace_runs:
        dotted(run)
    return orders


def load_saved(path: Path = SAVED) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _with_commit(extra: dict, commit: str | None) -> dict:
    """`extra` with its Context Hub commit replaced by `commit`, when both exist.

    The saved traces were written by the reference workspace's first commit, which does not
    exist in an attendee's repo. Their own first commit holds the same text, so replayed
    traces name that one, and the trace, Context Hub, and Engine agree on the version.
    """
    metadata = extra.get("metadata")
    if not commit or not isinstance(metadata, dict) or COMMIT_KEY not in metadata:
        return extra
    return {**extra, "metadata": {**metadata, COMMIT_KEY: commit}}


def upload(client, project: str, *, seed: int | None = None, tag: str | None = None,
           commit: str | None = None, log=print) -> dict:
    """Replay the saved traces into `project`. Returns counts, after checking they landed.

    `commit` is the workspace's own Context Hub commit for the bundled instructions; see
    `_with_commit`.
    """
    from langsmith import uuid7

    rng = random.Random(seed)
    runs = load_saved()
    id_map: dict[str, str] = {}
    for run in runs:
        if run.get("parent_run_id") is None:
            new = str(uuid7())
            id_map[str(run["id"])] = new
            id_map[str(run["trace_id"])] = new
    for run in runs:
        for field in ("id", "parent_run_id"):
            old = run.get(field)
            if old and str(old) not in id_map:
                id_map[str(old)] = str(uuid7())

    traces: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        trace_id = id_map[str(run["trace_id"])]
        traces[trace_id].append({
            "id": id_map[str(run["id"])],
            "trace_id": trace_id,
            "parent_run_id": (id_map[str(run["parent_run_id"])]
                              if run.get("parent_run_id") else None),
            "name": run["name"],
            "run_type": run["run_type"],
            "inputs": run.get("inputs") or {},
            "outputs": run.get("outputs"),
            "error": run.get("error"),
            "extra": _with_commit(run.get("extra") or {}, commit),
            "tags": [*(run.get("tags") or []),
                     *([tag] if tag and not run.get("parent_run_id") else [])] or None,
            "start_time": _parse(run["start_time"]),
            "end_time": _parse(run.get("end_time")),
            "feedback": [f for f in (run.get("feedback") or []) if f.get("key") == RATING_KEY],
        })

    now = datetime.now(UTC).replace(tzinfo=None)
    window = timedelta(days=MAX_BACKDATE_DAYS)
    for trace_runs in traces.values():
        start = min(r["start_time"] for r in trace_runs)
        end = max([r["end_time"] for r in trace_runs if r["end_time"]] or [start])
        latest = max(window - (end - start), timedelta(0))
        delta = (now - window + timedelta(seconds=rng.uniform(0, latest.total_seconds()))) - start
        for r in trace_runs:
            r["start_time"] += delta
            if r["end_time"]:
                r["end_time"] += delta

    from langsmith.utils import LangSmithNotFoundError

    try:
        project_id = client.read_project(project_name=project).id
    except LangSmithNotFoundError:
        project_id = client.create_project(project_name=project).id

    for trace_runs in traces.values():
        trace_runs.sort(key=lambda r: (r["parent_run_id"] is not None, r["start_time"]))
        orders = _dotted_orders(trace_runs)
        for run in trace_runs:
            client.create_run(
                id=run["id"], trace_id=run["trace_id"], dotted_order=orders[run["id"]],
                parent_run_id=run["parent_run_id"], name=run["name"],
                run_type=run["run_type"], inputs=run["inputs"], outputs=run["outputs"],
                error=run["error"], extra=run["extra"], tags=run["tags"],
                start_time=run["start_time"], end_time=run["end_time"],
                project_name=project,
            )
    client.flush()

    expected = {r["id"] for trace_runs in traces.values() for r in trace_runs}
    landed: set[str] = set()
    for _ in range(20):
        landed = {
            str(r.id) for r in client.list_runs(
                project_id=project_id, run_ids=list(expected),
                select=["id", "name", "start_time", "run_type", "trace_id"],
            )
        }
        if len(landed) == len(expected):
            break
        time.sleep(3)

    ratings = 0
    for trace_runs in traces.values():
        root = next(r for r in trace_runs if r["parent_run_id"] is None)
        found = root["feedback"] or []
        if not found:
            rated = rate({"outputs": root["outputs"]}, rng)
            if rated:
                found = [{"key": RATING_KEY, "score": rated[0], "comment": rated[1]}]
        for item in found:
            client.create_feedback(run_id=root["id"], trace_id=root["trace_id"], key=RATING_KEY,
                                   score=item.get("score"), comment=item.get("comment"))
            ratings += 1
    client.flush()
    result = {"traces": len(traces), "runs": len(expected), "landed": len(landed),
              "ratings": ratings, "project": project, "project_id": str(project_id)}
    log(f"saved traces: {len(traces)} traces, {len(landed)}/{len(expected)} runs landed, "
        f"{ratings} ratings, in {project}")
    return result


def main() -> int:
    from langsmith import Client

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["capture", "upload"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--limit", type=int, default=40, help="capture: root traces to keep")
    parser.add_argument("--since", type=datetime.fromisoformat, default=None,
                        help="capture: only traces started at or after this ISO time")
    args = parser.parse_args()
    client = Client()
    if args.action == "capture":
        capture(client, args.project, limit_traces=args.limit, since=args.since)
    else:
        upload(client, args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
