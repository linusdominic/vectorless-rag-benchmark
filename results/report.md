Results over 60 questions.

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Answer accuracy (95% CI) | Retrieval latency p50 | Retrieval cost per query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 70% | 82% | 92% | 0.78 | 85% (74% to 92%) | 1 ms | $0 |
| dense | 72% | 87% | 90% | 0.80 | 88% (78% to 94%) | 7 ms | $0 |
| hybrid | 75% | 90% | 92% | 0.82 | 92% (82% to 96%) | 8 ms | $0 |
| hybrid+parents | 75% | 90% | 92% | 0.82 | 87% (76% to 93%) | 8 ms | $0 |
| vectorless | 82% | 92% | 95% | 0.87 | 92% (82% to 96%) | 3.0 s | $0.00050 |
| vectorless+summaries | 95% | 100% | 100% | 0.97 | 98% (91% to 100%) | 2.9 s | $0.00127 |

| Method | Right section in top 3 | Said "not found" | Latency p90 | Retrieval tokens per query | Context sent to the answer model |
| --- | --- | --- | --- | --- | --- |
| bm25 | 87% | 15% | 1 ms | 0 | 5,125 chars |
| dense | 95% | 8% | 10 ms | 0 | 5,092 chars |
| hybrid | 97% | 5% | 10 ms | 0 | 5,097 chars |
| hybrid+parents | 97% | 8% | 10 ms | 0 | 11,911 chars |
| vectorless | 95% | 7% | 3.4 s | 3,899 | 8,606 chars |
| vectorless+summaries | 100% | 0% | 3.3 s | 11,587 | 8,187 chars |

By question style:

| Method | direct: Hit@3 | direct: answer accuracy | scenario: Hit@3 | scenario: answer accuracy |
| --- | --- | --- | --- | --- |
| bm25 | 97% | 93% | 67% | 77% |
| dense | 97% | 97% | 77% | 80% |
| hybrid | 97% | 97% | 83% | 87% |
| hybrid+parents | 97% | 93% | 83% | 80% |
| vectorless | 97% | 97% | 87% | 87% |
| vectorless+summaries | 100% | 100% | 100% | 97% |

Where the vectorless methods missed:

| Method | Found the gold paragraph | Missed at step 1 (wrong sections) | Missed at step 2 (right section, wrong paragraphs) |
| --- | --- | --- | --- |
| vectorless | 57 | 2 | 1 |
| vectorless+summaries | 60 | 0 | 0 |

Paired comparison with hybrid search (exact McNemar test on the same questions):

| Method vs hybrid | Hit@1: better / worse | Hit@1: p | Answer: better / worse | Answer: p |
| --- | --- | --- | --- | --- |
| bm25 | 2 / 5 | 0.453 | 2 / 6 | 0.289 |
| dense | 4 / 6 | 0.754 | 0 / 2 | 0.500 |
| hybrid+parents | 0 / 0 | 1.000 | 0 / 3 | 0.250 |
| vectorless | 10 / 6 | 0.454 | 5 / 5 | 1.000 |
| vectorless+summaries | 13 / 1 | 0.002 | 5 / 1 | 0.219 |
