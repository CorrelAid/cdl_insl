

## Current suggested pipeline to count verbs

- Spelling correction with LLMS -> POS tagging with stanza -> Count words tagged with VERB

## Thoughts

- Difference between spelling and grammatical mistakes, but choosing the correct word form depends on grammar
- Alternatives for spelling correction: https://github.com/wolfgarbe/SymSpell
- To evaulate performance of pipeline, some manually labeled data is necessary
    - manual POS tagging could be similar to: https://labelstud.io/templates/named_entity or https://labelstud.io/templates/relation_extraction
    - word spelling corrected could be pre-labeled by using a dictionary to find unknown words (however does not work for word form/grammatical mistakes)
- Everything depends on spelling correction

## Setup

1. Install uv
2. Install dependencies:
    ```
    uv sync
    ```
3. Optionally download spacy dependencies:
   ```
    uv run spacy download de_core_news_lg
    ```
