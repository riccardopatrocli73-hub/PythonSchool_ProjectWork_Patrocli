# Dictionary data sources

`it_bigrams.txt` and `en_bigrams.txt` contain aggregate adjacent-word counts
generated from the Italian and English Tatoeba sentence exports downloaded on
2026-10-01:

- https://downloads.tatoeba.org/exports/per_language/ita/ita_sentences.tsv.bz2
- https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2

The source sentences are contributed by Tatoeba users and distributed under
CC BY 2.0 FR. Author and sentence information is available from Tatoeba using
the sentence identifiers in the original exports. The generated dictionaries
do not contain the original sentences.

- Corpus information: https://tatoeba.org/en/downloads
- License: https://creativecommons.org/licenses/by/2.0/fr/deed.en

Generation command:

```bash
python -m src.build_bigram_dictionary INPUT.tsv.bz2 OUTPUT.txt \
  --minimum-count 2 --maximum-bigrams 200000
```
