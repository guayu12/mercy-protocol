import sys
"""Generate report-eval outputs for one chain by running real turns through process_query.

usage: venv/bin/python generate.py --chain luna|terra --runs 2 --out DIR   (cwd = backend/)
"""
import argparse, asyncio, json, os, secrets, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--chain", choices=["luna", "terra"], required=True)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--set", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_eval_set.json"))
ap.add_argument("--only", default="")
args = ap.parse_args()

# Must be set before importing the graph (load_dotenv does not override existing vars).
os.environ["MODEL_REPORT"] = "" if args.chain == "luna" else "openai:gpt-6.1-terra"
os.environ["MODEL_REPORT_FALLBACK"] = ""
# Isolate the report path: only MODEL_REPORT differs between chains.
for _var in ("MODEL_RESEARCH", "MODEL_RESEARCH_MAX", "MODEL_VISION", "OPENAI_VISION_MODEL"):
    os.environ[_var] = ""
sys.path.insert(0, os.getcwd())

from core import log_bus, media_store, session_store  # noqa: E402
from core.agent import graph  # noqa: E402


async def run_turn(prompt: str, thread_id: str) -> dict:
    queue = log_bus.subscribe()
    reports, order, spoken, tools, errors = {}, [], [], [], []
    started = time.time()
    try:
        async for event in graph.process_query(prompt, thread_id, None, {}, None, []):
            kind = event.get("type")
            if kind == "report.open":
                reports[event["report_id"]] = {"title": event.get("title", ""), "markdown": ""}
                order.append(event["report_id"])
            elif kind == "report.delta":
                reports.setdefault(event["report_id"], {"title": "", "markdown": ""})
                reports[event["report_id"]]["markdown"] += event.get("markdown", "")
            elif kind == "speech.sentence":
                spoken.append(event.get("text", ""))
            elif kind == "tool" and event.get("status") == "start":
                tools.append(event.get("name"))
            elif kind == "error":
                errors.append(event.get("message", ""))
    except Exception as exc:  # record and continue with the next prompt
        errors.append(f"{type(exc).__name__}: {exc}"[:300])
    latency = round(time.time() - started, 1)
    llm_logs = []
    while not queue.empty():
        entry = json.loads(queue.get_nowait())
        if entry.get("category") in {"LLM", "EXPERT"}:
            llm_logs.append(entry.get("data", {}))
    log_bus.unsubscribe(queue)
    return {
        "thread_id": thread_id,
        "latency_s": latency,
        "tools": tools,
        "spoken_text": "".join(spoken),
        "reports": [reports[r] | {"report_id": r} for r in order],
        "llm_calls": llm_logs,
        "errors": errors,
    }


async def main():
    data = json.load(open(args.set))
    os.makedirs(args.out, exist_ok=True)
    await session_store.init_store(os.path.join(args.out, f"sessions_{args.chain}.db"))
    await media_store.init_store(os.path.join(args.out, f"sessions_{args.chain}.db"))
    await graph.init_graph()
    bound = sorted(t.name for t in graph.tools)
    results = []
    for prompt in data["prompts"]:
        if args.only and prompt["id"] not in args.only.split(","):
            continue
        for run in range(1, args.runs + 1):
            thread_id = f"eval_{args.chain}_{prompt['id']}_{run}_{secrets.token_hex(3)}"
            result = await run_turn(prompt["prompt"], thread_id)
            result |= {"id": prompt["id"], "run": run, "chain": args.chain}
            results.append(result)
            print(f"{args.chain} {prompt['id']} run{run}: {result['latency_s']}s tools={result['tools']} "
                  f"reports={len(result['reports'])} chars={sum(len(r['markdown']) for r in result['reports'])} "
                  f"errors={len(result['errors'])}", flush=True)
            json.dump({"chain": args.chain, "bound_tools": bound, "results": results},
                      open(os.path.join(args.out, f"outputs_{args.chain}.json"), "w"), ensure_ascii=False, indent=2)
    await graph.close_graph()


asyncio.run(main())

sys.stdout.flush()
os._exit(0)
