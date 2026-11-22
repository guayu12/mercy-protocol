# Report comparison results (2026-11-22, PR #9)

Setup: 10 prompts × 2 runs per chain. Both chains had research/vision disabled; only `MODEL_REPORT` differed
(luna: empty → Luna writes with `show_report`; terra: `openai:gpt-6.1-terra` → Luna may delegate via `write_report`).
Judge: gpt-6.2-sol, blind, randomized A/B, reference answers for objective prompts.

| Metric | Luna | Terra chain |
|---|---|---|
| Correctness mean (all 20 pairs) | 4.25 | 4.80 |
| Prompts with mean lead ≥ 0.5 (rule a) | — | 5/10 (fails) |
| Prompts where Luna had errors and Terra none (rule c) | — | 4 (passes) |

Rule b (correctness lead ≥ 0.5) passes, so `keep_terra = true`.

Caveats
- Luna did not delegate in 5 of 20 terra-chain runs (r01 ×2, r04 ×1, r07 ×2); both sides were Luna there.
  Correctness differences on those pairs were −1, +1, 0, 0, +1 (correctness means 4.6 vs 4.8; five-criterion means 4.6 vs 4.6):
  single-pair judge noise is about ±1.
- Delegated pairs only (15): Terra correctness higher in 9, equal in 6, lower in 0; mean 4.80 vs 4.13.
- Position bias check (delegated pairs): Terra lead is +0.50 when shown as A (6 pairs) and +0.78 as B (9 pairs).
- Cost: delegated runs took 43.2 s on average vs 15.1 s for Luna on the same prompts (max 83.8 s).
  Median report length 3534 vs 1941 characters; the longest Terra report was 13640 characters,
  above the 12000-character limit returned to Luna (she is told to use `read_report`).

Raw outputs (`outputs_*.json`, `judgments.json`, `summary.json`) were shared with the user and the advisor and are not committed.
