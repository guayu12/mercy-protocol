"""Blind pairwise judging of luna vs terra outputs with gpt-6.2-sol (reasoning high).

usage: venv/bin/python judge.py DIR   (reads DIR/outputs_luna.json, DIR/outputs_terra.json; writes DIR/judgments.json, DIR/summary.json)
"""
import asyncio, json, os, random, sys
from dotenv import load_dotenv
from openai import AsyncOpenAI

out = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.getcwd(), ".env"))
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None)
JUDGE_MODEL = "gpt-6.2-sol"
CRITERIA = ["correctness", "completeness", "structure", "specificity", "language"]

JUDGE_PROMPT = """你是严格、公正的评审。下面是用户的请求和两份由不同系统生成、显示在屏幕上的回答（A 和 B）。请分别独立评价。
评分项（每项 1–5 分，5 最好）：
- correctness 正确性：事实、数字、代码和命令是否正确；给了参考要点时以参考要点为准；编造来源、API、数据或命令要重扣。
- completeness 完整性：是否覆盖请求的每个部分。
- structure 结构：标题、表格、列表是否合适，在平板上是否易读。
- specificity 具体性：是否给出具体步骤、数字和示例，而不是空泛的话。
- language 语言：简体中文是否自然、术语是否一致。
另外，逐条列出每份回答里的事实错误或计算错误；没有就给空列表。只算确实错误的内容，不算风格偏好。
不要因为篇幅长或格式花哨就给高分；A 和 B 的先后顺序与质量无关。
只输出 JSON，格式：{"A": {"correctness": n, "completeness": n, "structure": n, "specificity": n, "language": n, "errors": ["..."]}, "B": {...}}"""


def answer_text(result: dict) -> str:
    if result["reports"]:
        return "\n\n".join(f"# {r['title']}\n\n{r['markdown']}" for r in result["reports"])
    return result["spoken_text"] or "（没有输出）"


async def judge_pair(prompt: dict, a: str, b: str) -> dict:
    user = f"用户请求：\n{prompt['prompt']}\n\n"
    if prompt.get("reference"):
        user += f"参考要点：\n{prompt['reference']}\n\n"
    user += f"=== 回答 A ===\n{a}\n\n=== 回答 B ===\n{b}"
    for attempt in range(3):
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            reasoning_effort="high",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
        )
        try:
            parsed = json.loads(resp.choices[0].message.content)
            for side in ("A", "B"):
                for c in CRITERIA:
                    assert 1 <= int(parsed[side][c]) <= 5
                assert isinstance(parsed[side]["errors"], list)
            return parsed
        except Exception:
            if attempt == 2:
                raise


async def main():
    data = json.load(open(os.path.join(here, "report_eval_set.json")))
    prompts = {p["id"]: p for p in data["prompts"]}
    luna = {(r["id"], r["run"]): r for r in json.load(open(os.path.join(out, "outputs_luna.json")))["results"]}
    terra = {(r["id"], r["run"]): r for r in json.load(open(os.path.join(out, "outputs_terra.json")))["results"]}
    rng = random.Random(20261122)
    judgments = []
    for key in sorted(set(luna) & set(terra)):
        pid, run = key
        swap = rng.random() < 0.5
        first, second = (terra[key], luna[key]) if swap else (luna[key], terra[key])
        verdict = await judge_pair(prompts[pid], answer_text(first), answer_text(second))
        mapping = {"A": "terra" if swap else "luna", "B": "luna" if swap else "terra"}
        judgments.append({"id": pid, "run": run, "order": mapping,
                          "luna": verdict["A" if mapping["A"] == "luna" else "B"],
                          "terra": verdict["A" if mapping["A"] == "terra" else "B"]})
        print(pid, run, {m: round(sum(judgments[-1][m][c] for c in CRITERIA) / 5, 2) for m in ("luna", "terra")}, flush=True)
        json.dump(judgments, open(os.path.join(out, "judgments.json"), "w"), ensure_ascii=False, indent=2)

    per_prompt = {}
    for pid in prompts:
        rows = [j for j in judgments if j["id"] == pid]
        if not rows:
            continue
        stat = {}
        for m in ("luna", "terra"):
            stat[m] = {
                "mean": round(sum(sum(r[m][c] for c in CRITERIA) / 5 for r in rows) / len(rows), 3),
                "correctness": round(sum(r[m]["correctness"] for r in rows) / len(rows), 3),
                "has_errors": any(r[m]["errors"] for r in rows),
                "latency_s": round(sum((luna if m == "luna" else terra)[(pid, r["run"])]["latency_s"] for r in rows) / len(rows), 1),
                "tools": [(luna if m == "luna" else terra)[(pid, r["run"])]["tools"] for r in rows],
            }
        per_prompt[pid] = stat
    n = len(per_prompt)
    wins = sum(1 for s in per_prompt.values() if s["terra"]["mean"] >= s["luna"]["mean"] + 0.5)
    corr_l = sum(s["luna"]["correctness"] for s in per_prompt.values()) / n
    corr_t = sum(s["terra"]["correctness"] for s in per_prompt.values()) / n
    luna_only_errors = sum(1 for s in per_prompt.values() if s["luna"]["has_errors"] and not s["terra"]["has_errors"])
    summary = {
        "prompts_judged": n,
        "terra_wins_by_0.5": wins,
        "correctness_mean": {"luna": round(corr_l, 3), "terra": round(corr_t, 3)},
        "prompts_luna_errors_terra_clean": luna_only_errors,
        "rule_a": wins >= 6, "rule_b": corr_t >= corr_l + 0.5, "rule_c": luna_only_errors >= 2,
        "per_prompt": per_prompt,
    }
    summary["keep_terra"] = summary["rule_a"] or summary["rule_b"] or summary["rule_c"]
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, ensure_ascii=False))


asyncio.run(main())
