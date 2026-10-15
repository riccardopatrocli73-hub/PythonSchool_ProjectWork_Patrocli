"""OCR recognized word correction"""

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from symspellpy import SymSpell, Verbosity

from src.alphabet import LETTERS

# Wheights used for ranking
EDIT_DISTANCE_WEIGHT = 6.0  # Penalizes words far to read one
OCR_LOG_PROBABILITY_WEIGHT = 1.0  # Reward OCR likely words
DICTIONARY_FREQUENCY_WEIGHT = 0.20  # Reward frequent words
CHARACTER_ORDER_WEIGHT = 3.0  # Reward similar words
LENGTH_DIFFERENCE_WEIGHT = 0.25  # Penalizes different length words


# OCR word definition
@dataclass(frozen=True)
class OCRCandidate:
    word: str
    log_probability: float

    @property
    def confidence(self):
        if not self.word:
            return 0.0

        return math.exp(self.log_probability / len(self.word))


# Represents a corrector suggested word
@dataclass(frozen=True)
class SpellingSuggestion:
    word: str
    edit_distance: int
    frequency: int
    ocr_log_probability: float


# Final correction result
@dataclass(frozen=True)
class CorrectionResult:
    raw_word: str
    is_correct: bool
    correction: str | None
    suggestions: list[SpellingSuggestion]


# Word normalization to compare with dictionary
def normalize_word(word):
    normalized = unicodedata.normalize(
        "NFKD",
        word.casefold(),
    )

    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )

    return re.sub(r"[^a-z]", "", normalized)


# Different words building from model top-k predicitons
def word_edit_distance(left, right):
    """Levenshtein distance between two normalized spellings."""
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def build_ocr_candidates(
    top_probabilities,
    top_labels,
    beam_width=50,
):
    """
    --------------------------------------------------------------------
    This function uses a beam type search:
    1) It begins from an empty candidate
    2) For each position it expands the existing candidates
       with all possible letters
    3) It uses log-probabilities to use sum instead of multiplication
    4) After each ch, cnadidates are sorted
    5) Returns best OCR words possibilities
    --------------------------------------------------------------------
    """
    beams = [OCRCandidate("", 0.0)]

    for probabilities, labels in zip(top_probabilities, top_labels):
        expanded = []

        for candidate in beams:
            for probability, label in zip(probabilities, labels):
                probability = max(
                    float(probability),
                    1e-12,
                )

                expanded.append(
                    OCRCandidate(
                        word=(candidate.word + LETTERS[int(label)]),
                        log_probability=(
                            candidate.log_probability + math.log(probability)
                        ),
                    )
                )

        expanded.sort(
            key=lambda candidate: candidate.log_probability,
            reverse=True,
        )

        beams = expanded[:beam_width]

    return beams


# Corrector
class ItalianSpellCorrector:
    def __init__(
        self,
        dictionary_path,
        max_edit_distance=2,
        custom_dictionary_path=None,
        extra_dictionary_paths=None,
        custom_frequency=10_000_000,
    ):
        self.dictionary_path = Path(dictionary_path)
        self.custom_dictionary_path = (
            Path(custom_dictionary_path) if custom_dictionary_path else None
        )
        self.extra_dictionary_paths = [
            Path(dictionary_path) for dictionary_path in (extra_dictionary_paths or [])
        ]
        self.custom_frequency = custom_frequency
        self.max_edit_distance = max_edit_distance

        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {self.dictionary_path}")

        optional_dictionaries = [
            path
            for path in [self.custom_dictionary_path, *self.extra_dictionary_paths]
            if path is not None
        ]

        for dictionary_path in optional_dictionaries:
            if not dictionary_path.exists():
                raise FileNotFoundError(f"Dictionary not found: {dictionary_path}")

        # SymSpell finds suggestions similar to a wrong word
        self.engine = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )

        # Creation of internal dictionaries
        self.surface_forms = {}
        self.surface_frequencies = {}

        self._load_dictionary()

    # Dictionary word addition
    def _add_dictionary_entry(self, surface, frequency):
        normalized = normalize_word(surface)

        if not normalized:
            return

        self.engine.create_dictionary_entry(
            normalized,
            frequency,
        )

        previous_frequency = self.surface_frequencies.get(
            normalized,
            -1,
        )

        if frequency > previous_frequency:
            self.surface_forms[normalized] = surface
            self.surface_frequencies[normalized] = frequency

    # Load dictionary in format (word, frequency)
    def _load_weighted_dictionary(self, dictionary_path):
        with dictionary_path.open(
            encoding="utf-8",
        ) as dictionary:
            for line in dictionary:
                parts = line.strip().rsplit(maxsplit=1)

                if len(parts) != 2:
                    continue

                surface, frequency_text = parts

                try:
                    frequency = int(frequency_text)
                except ValueError:
                    continue

                self._add_dictionary_entry(surface, frequency)

    def _load_custom_dictionary(self, dictionary_path):
        with dictionary_path.open(
            encoding="utf-8",
        ) as dictionary:
            for line in dictionary:
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                parts = stripped.rsplit(maxsplit=1)
                surface = stripped
                frequency = (
                    self.custom_frequency
                )  # custom_frequency = 10000000 -> Custom words are more relevant

                if len(parts) == 2:
                    try:
                        frequency = int(parts[1])
                        surface = parts[0]
                    except ValueError:
                        pass

                self._add_dictionary_entry(surface, frequency)

    def _load_dictionary(self):
        self._load_weighted_dictionary(self.dictionary_path)

        for dictionary_path in self.extra_dictionary_paths:
            self._load_weighted_dictionary(dictionary_path)

        if self.custom_dictionary_path is not None:
            self._load_custom_dictionary(self.custom_dictionary_path)

        if not self.engine.words:
            raise ValueError(f"Empty dictionary: {self.dictionary_path}")

    # This function returns original word, if exists
    def _surface_form(self, normalized):
        return self.surface_forms.get(
            normalized,
            normalized,
        )

    # Longest Common Subsequence
    def _lcs_length(self, left, right):
        previous = [0] * (len(right) + 1)

        for left_character in left:
            current = [0]

            for index, right_character in enumerate(right, start=1):
                if left_character == right_character:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[index - 1]))

            previous = current

        return previous[-1]

    # This function assigns a similarity score between OCR word and suggestion
    def _visual_similarity_score(self, suggestion, normalized_raw):
        normalized_suggestion = normalize_word(suggestion.word)

        if not normalized_raw or not normalized_suggestion:
            return 0.0

        lcs_ratio = self._lcs_length(
            normalized_raw,
            normalized_suggestion,
        ) / len(normalized_raw)
        length_difference = abs(len(normalized_suggestion) - len(normalized_raw))
        length_penalty = length_difference / max(
            len(normalized_raw),
            len(normalized_suggestion),
        )

        return (CHARACTER_ORDER_WEIGHT * lcs_ratio) - (
            LENGTH_DIFFERENCE_WEIGHT * length_penalty
        )

    # This function returns suggestions scores
    def _ranking_score(self, suggestion, normalized_raw=None):
        score = (
            (OCR_LOG_PROBABILITY_WEIGHT * suggestion.ocr_log_probability)
            + (DICTIONARY_FREQUENCY_WEIGHT * math.log1p(suggestion.frequency))
            - (EDIT_DISTANCE_WEIGHT * suggestion.edit_distance)
        )

        if normalized_raw is not None:
            score += self._visual_similarity_score(suggestion, normalized_raw)

        return score

    # This function decides if a suggestion is reliable enough
    def _is_reliable(self, suggestion, normalized_raw):
        if suggestion.edit_distance == 0:
            return True

        comparison_length = max(
            len(normalized_raw),
            len(normalize_word(suggestion.word)),
        )

        if comparison_length < 3:
            return False

        allowed_distance = max(
            1,
            math.floor(comparison_length * 0.40),
        )

        return suggestion.edit_distance <= min(
            self.max_edit_distance,
            allowed_distance,
        )

    # Main function
    def correct(
        self,
        raw_word,  # Best model read word
        ocr_candidates,  # Alternatives
        suggestion_limit=3,
    ):
        normalized_raw = normalize_word(raw_word)
        raw_is_correct = (
            normalized_raw in self.engine.words
        )  # Check if raw_word in dict

        candidate_scores = {}

        for candidate in ocr_candidates:
            normalized = normalize_word(candidate.word)

            if not normalized:
                continue

            previous = candidate_scores.get(
                normalized,
                float("-inf"),
            )

            candidate_scores[normalized] = max(
                previous,
                candidate.log_probability,
            )

        collected = {}  # Set of possible suggestions

        for normalized, ocr_score in candidate_scores.items():
            if normalized in self.engine.words:
                frequency = self.engine.words[normalized]

                # Valid words OCR candidates added to dictionary
                collected[normalized] = SpellingSuggestion(
                    word=self._surface_form(normalized),
                    edit_distance=0,
                    frequency=frequency,
                    ocr_log_probability=ocr_score,
                )

        for normalized, ocr_score in candidate_scores.items():
            # OCR asks SymSpell similar words for each candidate
            suggestions = self.engine.lookup(
                normalized,
                Verbosity.ALL,
                max_edit_distance=self.max_edit_distance,
                include_unknown=False,
            )

            for suggestion in suggestions:
                existing = collected.get(suggestion.term)

                candidate = SpellingSuggestion(
                    word=self._surface_form(suggestion.term),
                    edit_distance=suggestion.distance,
                    frequency=suggestion.count,
                    ocr_log_probability=ocr_score,
                )

                # If a word already has a suggestion, the one with highest score is kept
                if existing is None or self._ranking_score(
                    candidate,
                    normalized_raw,
                ) > self._ranking_score(
                    existing,
                    normalized_raw,
                ):
                    collected[suggestion.term] = candidate

        ranked = sorted(
            collected.values(),
            key=lambda suggestion: self._ranking_score(
                suggestion,
                normalized_raw,
            ),
            reverse=True,
        )

        ranked = ranked[:suggestion_limit]

        # Final decision
        if raw_is_correct:
            correction = self._surface_form(normalized_raw)
        elif ranked and self._is_reliable(
            ranked[0],
            normalized_raw,
        ):
            correction = ranked[0].word
        else:
            correction = None

        return CorrectionResult(
            raw_word=raw_word,
            is_correct=raw_is_correct,
            correction=correction,
            suggestions=ranked,
        )
