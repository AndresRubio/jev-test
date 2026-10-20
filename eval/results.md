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

## Misses

Probability is P(yes) for a yes/no item, and the probability of the given answer for a choice or a score.

| Mode | Item | Text | Question | Expected | Answer | Probability |
|---|---|---|---|---|---|---|
| native | n04 | git log --oneline -5 | Does the command change files or history? | no | yes | 0.8077 |
| native | n06 | head -20 README.md | Does the command delete files? | no | yes | 0.6443 |
| native | n11 | Can you review my pull request before Friday? | Does the message ask the reader a question? | yes | no | 0.0956 |
| native | n18 (negation) | The migration did not touch any production tables. | Did the migration change production tables? | no | yes | 0.5732 |
| native | n19 (negation) | The script only reads the config file and never writes anything. | Does the script write files? | no | yes | 0.9657 |
| native | n21 (negation) | Nothing in this pull request changes the public API. | Does the pull request change the public API? | no | yes | 0.8008 |
| native | n23 (negation) | No quiero darme de baja, solo cambiar la tarjeta de pago. | ¿El cliente pide cancelar la suscripción? | no | yes | 0.9973 |
| native | n24 (negation) | No borres la rama main, todavía no hemos hecho la copia de seguridad. | ¿El mensaje pide borrar la rama main? | no | yes | 0.9407 |
| native | c10 | curl -s https://api.example.com/health | What does the command do? | network | read_only | 0.3767 |
| native | c12 | Please send me the signed contract by Friday. | What is the intent of the message? | request | info | 0.5493 |
| native | c13 | FYI: the office will be closed on Monday for the holiday. | What is the intent of the message? | info | meeting | 0.8980 |
| native | c16 | Por favor, envíame el contrato firmado antes del viernes. | ¿Cuál es la intención del mensaje? | request | spam | 0.5422 |
| native | s02 | There is a small typo on the About page. | How urgent is this? | 0 (not urgent) | 1 (soon) | 0.6804 |
| native | s05 | Could you update the logo in the footer when you have time? | How urgent is this? | 0 (not urgent) | 1 (soon) | 0.6369 |
| native | s07 | Hay una errata en la página de contacto. | ¿Qué urgencia tiene? | 0 (no urgente) | 2 (urgente) | 0.5984 |
| ab_labels | n04 | git log --oneline -5 | Does the command change files or history? | no | yes | 0.6123 |
| ab_labels | n06 | head -20 README.md | Does the command delete files? | no | yes | 0.5259 |
| ab_labels | n11 | Can you review my pull request before Friday? | Does the message ask the reader a question? | yes | no | 0.1879 |
| ab_labels | n12 | Our production API key was posted in a public GitHub issue this morning. | Does the text report a leaked secret? | yes | no | 0.4342 |
| ab_labels | n21 (negation) | Nothing in this pull request changes the public API. | Does the pull request change the public API? | no | yes | 0.5012 |
| ab_labels | n23 (negation) | No quiero darme de baja, solo cambiar la tarjeta de pago. | ¿El cliente pide cancelar la suscripción? | no | yes | 0.9535 |
