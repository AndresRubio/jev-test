# Evaluation (2026-10-20)

60 hand-labelled items (`eval/items.jsonl`), 12 of them in Spanish, one question per request.

## noul_mode = native

| Slice | Correct | Accuracy |
|---|---|---|
| choice | 16/20 | 80% |
| choice · english | 13/16 | 81% |
| choice · multilingual | 3/4 | 75% |
| noul | 22/30 | 73% |
| noul · english | 18/24 | 75% |
| noul · multilingual | 4/6 | 67% |
| noul · negations | 5/10 | 50% |
| score | 7/10 | 70% |
| score · english | 6/8 | 75% |
| score · multilingual | 1/2 | 50% |

## noul_mode = ab_labels

| Slice | Correct | Accuracy |
|---|---|---|
| noul | 24/30 | 80% |
| noul · english | 19/24 | 79% |
| noul · multilingual | 5/6 | 83% |
| noul · negations | 8/10 | 80% |

The default `noul_mode` is `ab_labels`, the mode with more correct yes/no answers: native 22/30, ab_labels 24/30. A tie goes to `native`.
