""" Compact word-bigram models used to score short OCR phrases. """

import math
from functools import lru_cache
from pathlib import Path

from src.spell_correct import normalize_word

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIGRAM_DICTIONARIES = {
    "Italian": PROJECT_ROOT / "data" / "dictionaries" / "it_bigrams.txt",
    "English": PROJECT_ROOT / "data" / "dictionaries" / "en_bigrams.txt",
}


class BigramLanguageModel:
    """Read-only frequency model for adjacent normalized words."""

    def __init__(self, counts):
        normalized_counts = {}

        for (left, right), count in counts.items():
            normalized_left = normalize_word(left)
            normalized_right = normalize_word(right)
            count = int(count)

            if normalized_left and normalized_right and count > 0:
                normalized_counts[(normalized_left, normalized_right)] = count

        self.counts = normalized_counts

    @classmethod
    def from_path(cls, path):
        """
        Bigrams file reading; each row is divided in three parts 
        and returns an obj BigramLanguageModel
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Bigram dictionary not found: {path}")

        counts = {}

        with path.open(encoding="utf-8") as bigrams:
            for line in bigrams:
                parts = line.split()

                if len(parts) != 3:
                    continue

                left, right, count_text = parts

                try:
                    count = int(count_text)
                except ValueError:
                    continue

                if count > 0:
                    counts[(left, right)] = count

        if not counts:
            raise ValueError(f"Empty bigram dictionary: {path}")

        return cls(counts)

    def pair_count(self, left, right):
        """
        This function returns how many time
        a couple appears
        """
        return self.counts.get(
            (normalize_word(left), normalize_word(right)),
            0,
        )

    def local_score(self, previous_word, word, next_word):
        """Return log-frequency evidence supplied by the available neighbours."""

        counts = []

        if previous_word:
            counts.append(self.pair_count(previous_word, word))

        if next_word:
            counts.append(self.pair_count(word, next_word))

        return sum(math.log1p(count) for count in counts)

    def has_local_evidence(self, previous_word, word, next_word):
        """
        Returns True if one of the two bigrams exists in the diciotnary
        """
        return bool(
            (previous_word and self.pair_count(previous_word, word))
            or (next_word and self.pair_count(word, next_word))
        )


@lru_cache(maxsize=None)
def _load_bigram_model(path):
    return BigramLanguageModel.from_path(path)


def load_bigram_models(paths=None):
    model_paths = DEFAULT_BIGRAM_DICTIONARIES if paths is None else paths
    models = {}

    for language, path in model_paths.items():
        path = Path(path)

        if path.exists():
            models[language] = _load_bigram_model(str(path.resolve()))

    return models
