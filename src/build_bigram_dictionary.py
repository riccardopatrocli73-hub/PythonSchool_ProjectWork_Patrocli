""" Build a compact word-bigram dictionary from a Tatoeba sentence export. """

import argparse
import bz2
import re
from collections import Counter
from pathlib import Path

from src.spell_correct import normalize_word

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)

# Ch sequence extraction and discard token longer than 40 letters
def tokenize(text):
    words = [normalize_word(match.group(0)) for match in WORD_PATTERN.finditer(text)]
    return [word for word in words if word and len(word) <= 40]

# TSV line reading
def read_sentences(path):
    path = Path(path)
    if path.suffix == ".bz2":
        source_context = bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    else:
        source_context = path.open(mode="rt", encoding="utf-8", errors="replace")

    with source_context as source:
        for line in source:
            parts = line.rstrip("\n").split("\t", maxsplit=2)

            if len(parts) == 3:
                yield parts[2]

# Bigrams count
def count_bigrams(sentences):
    counts = Counter()

    for sentence in sentences:
        words = tokenize(sentence)
        counts.update(zip(words, words[1:]))

    return counts

# Bigrams sorting based on frequency 
def write_bigrams(counts, output_path, minimum_count=2, maximum_bigrams=200_000):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = (
        (pair, count) for pair, count in counts.most_common() if count >= minimum_count
    )   # Not frequent bigrams are discarded

    with output_path.open("w", encoding="utf-8") as output:
        for index, ((left, right), count) in enumerate(selected):
            if index >= maximum_bigrams:
                break

            output.write(f"{left} {right} {count}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Tatoeba TSV or TSV.BZ2 export")
    parser.add_argument("output", type=Path, help="Destination bigram dictionary")
    parser.add_argument("--minimum-count", type=int, default=2)
    parser.add_argument("--maximum-bigrams", type=int, default=200_000)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    counts = count_bigrams(read_sentences(arguments.input))
    write_bigrams(
        counts,
        arguments.output,
        minimum_count=arguments.minimum_count,
        maximum_bigrams=arguments.maximum_bigrams,
    )


if __name__ == "__main__":
    main()
