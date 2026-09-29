# WealthGuard AI — RAG Evaluation Summary

## Evaluation Overview

- Test cases: 5
- Average correctness: 40.00%
- Average grounding: 80.00%
- Average citation quality: 80.00%
- Average source retrieval: 60.00%

## Evaluation Method

Correctness and grounding use expected-keyword checks.
Citation quality compares cited filenames with expected source files.
Source retrieval measures whether expected documents were retrieved.

These are automated proxy metrics, not a complete semantic or human evaluation.

## Notable Failures

- **Q2**: Correctness=0.00, Grounding=1.00, Citation=1.00. Review retrieved context, answer, and citation output. Possible causes include retrieval mismatch, missing source metadata, answer wording variation, or unsupported generation.
- **Q4**: Correctness=0.00, Grounding=0.00, Citation=0.00. Review retrieved context, answer, and citation output. Possible causes include retrieval mismatch, missing source metadata, answer wording variation, or unsupported generation.
- **Q5**: Correctness=0.00, Grounding=1.00, Citation=1.00. Review retrieved context, answer, and citation output. Possible causes include retrieval mismatch, missing source metadata, answer wording variation, or unsupported generation.

## Limitations

- Keyword matching may miss semantically correct paraphrases.
- A cited filename does not prove that every claim is supported.
- Similarity retrieval can return relevant-looking but insufficient context.
- Results depend on the current test set, database, and model response.
- Manual review is required before treating scores as final quality evidence.
