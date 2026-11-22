# Report comparison evaluation

From the repository root, generate two runs for each chain into the same output
directory, then judge the paired results:

```sh
cd backend
venv/bin/python eval/report_compare/generate.py --chain luna --runs 2 --out DIR
venv/bin/python eval/report_compare/generate.py --chain terra --runs 2 --out DIR
venv/bin/python eval/report_compare/judge.py DIR
```

Replace `DIR` with the desired output directory. The scripts call real model
APIs; generation and judging incur API costs. Judging uses `gpt-6.2-sol` with
high reasoning effort.

Latest results: [RESULTS.md](RESULTS.md)

Keep Terra only if at least one rule passes:

- **a.** Terra's mean across the five criteria is at least 0.5 above Luna's on
  at least 6 of the 10 prompts.
- **b.** Terra's mean correctness score is at least 0.5 above Luna's.
- **c.** On at least 2 prompts, Luna has a judged factual or calculation error
  and Terra has none.

The user makes the final decision after reviewing the raw outputs.
