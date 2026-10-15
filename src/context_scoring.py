"""Phrase-level language selection and conservative contextual scoring."""

import math
from dataclasses import replace

from src.spell_correct import normalize_word, word_edit_distance

# Costants used to avoid aggressive corrections
CONTEXTUAL_CONFIDENCE_THRESHOLD = 0.85
CONTEXTUAL_EDIT_PENALTY = 0.25
CONTEXTUAL_MINIMUM_GAIN = 0.35
CONTEXTUAL_BIGRAM_WEIGHT = 1.25
CONTEXTUAL_FREQUENCY_WEIGHT = 0.20
CONTEXTUAL_SUGGESTION_LIMIT = 25
CONTEXTUAL_SHORT_WORD_LENGTH = 4
CONTEXTUAL_BEAM_WIDTH = 50


def _phrase_language(word_results, language_correctors):
    """Prefer exact, unambiguous words over guesses from misspelled words."""
    votes = {}
    for result in word_results:
        word = normalize_word(result.raw_word)
        matches = [
            item["name"]
            for item in language_correctors
            if word in item["corrector"].engine.words
        ]
        if len(matches) == 1:
            votes[matches[0]] = votes.get(matches[0], 0) + 1
    ranked = sorted(votes, key=votes.get, reverse=True)
    if ranked and (len(ranked) == 1 or votes[ranked[0]] > votes[ranked[1]]):
        if votes[ranked[0]] >= 2 or (len(ranked) == 1 and len(word_results) >= 3):
            return ranked[0]
    # Ambiguous dictionary matches must not decide the phrase language.
    if votes:
        return None
    unambiguous = [
        result
        for result in word_results
        if " (also found in " not in result.detected_language
    ]
    return select_dominant_language(unambiguous)


# This funcion select the language of the phrase
def select_dominant_language(word_results):
    evidence = {}

    for result in word_results:
        primary_language = result.detected_language.split(
            " (also found in ",
            maxsplit=1,
        )[0]

        if primary_language not in {"Italian", "English"}:
            continue

        evidence[primary_language] = evidence.get(primary_language, 0) + 1

    if not evidence:
        return None

    ranked = sorted(evidence.items(), key=lambda item: item[1], reverse=True)

    if ranked[0][1] < 2:
        return None

    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None

    return ranked[0][0]


# This function takes previous or next word if on the same line
def _same_line_neighbour(word_results, index, offset):
    neighbour_index = index + offset

    if neighbour_index < 0 or neighbour_index >= len(word_results):
        return None

    current_predictions = word_results[index].character_predictions
    neighbour_predictions = word_results[neighbour_index].character_predictions

    if not current_predictions or not neighbour_predictions:
        return None

    if current_predictions[0].line_index != neighbour_predictions[0].line_index:
        return None

    return word_results[neighbour_index].final_word


# This function decides if a word has to be changed based on the context
def choose_contextual_suggestion(
    raw_word,
    predictions,
    suggestions,
    previous_word=None,
    next_word=None,
    language_model=None,
):
    """Choose a safer alternative for an uncertain valid dictionary word."""

    normalized_raw = normalize_word(raw_word)

    """
    No corrections if the word is short or
    no character is uncertain e probabilities are high
    """
    if len(normalized_raw) < 2 or not predictions:
        return None

    if (
        not any(prediction.uncertain for prediction in predictions)
        and min(prediction.best_probability for prediction in predictions)
        >= CONTEXTUAL_CONFIDENCE_THRESHOLD
    ):
        return None

    # Comparison between suggestions and original word
    raw_suggestion = next(
        (
            suggestion
            for suggestion in suggestions
            if normalize_word(suggestion.word) == normalized_raw
        ),
        None,
    )

    if raw_suggestion is None:
        return None

    # Check for context evidence for bigrams
    has_context = language_model is not None and any(
        language_model.has_local_evidence(previous_word, suggestion.word, next_word)
        for suggestion in suggestions
    )

    # If word is short or no context is detected, no corrections
    if len(normalized_raw) < CONTEXTUAL_SHORT_WORD_LENGTH and not has_context:
        return None

    frequency_weight = CONTEXTUAL_FREQUENCY_WEIGHT if has_context else 1.0
    raw_score = raw_suggestion.ocr_log_probability + (
        frequency_weight * math.log1p(raw_suggestion.frequency)
    )

    if has_context:
        raw_score += CONTEXTUAL_BIGRAM_WEIGHT * language_model.local_score(
            previous_word,
            raw_suggestion.word,
            next_word,
        )

    best_choice = None
    best_gain = CONTEXTUAL_MINIMUM_GAIN

    for suggestion in suggestions:
        normalized_suggestion = normalize_word(suggestion.word)
        suggestion_has_context = (
            language_model is not None
            and language_model.has_local_evidence(
                previous_word,
                suggestion.word,
                next_word,
            )
        )

        if (
            normalized_suggestion == normalized_raw
            or suggestion.edit_distance != 1
            or len(normalized_suggestion) != len(normalized_raw)
        ):
            continue

        if (
            len(normalized_raw) < CONTEXTUAL_SHORT_WORD_LENGTH
            and not suggestion_has_context
        ):
            continue

        candidate_score = (
            suggestion.ocr_log_probability
            + (frequency_weight * math.log1p(suggestion.frequency))
            - (CONTEXTUAL_EDIT_PENALTY * suggestion.edit_distance)
        )

        if has_context:
            candidate_score += CONTEXTUAL_BIGRAM_WEIGHT * language_model.local_score(
                previous_word,
                suggestion.word,
                next_word,
            )

        gain = candidate_score - raw_score

        # If gain is high, choose that suggestion
        if gain > best_gain:
            best_choice = suggestion
            best_gain = gain

    return best_choice


# This function applies the previous logic to the whole phrase
def apply_phrase_context(
    word_results,
    ocr_candidate_groups,
    language_correctors,
    bigram_models=None,
):
    """Apply language-consistent corrections to uncertain valid words."""

    if len(word_results) < 2:
        return word_results

    if len(word_results) != len(ocr_candidate_groups):
        raise ValueError("Each word requires its own OCR candidate group")

    dominant_language = _phrase_language(word_results, language_correctors)

    if dominant_language is None:
        return word_results

    # Find corrector
    language = next(
        (item for item in language_correctors if item["name"] == dominant_language),
        None,
    )

    if language is None:
        return word_results

    options = []
    language_model = (bigram_models or {}).get(dominant_language)

    for index, (result, ocr_candidates) in enumerate(
        zip(word_results, ocr_candidate_groups)
    ):
        normalized_raw = normalize_word(result.raw_word)
        exact_here = normalized_raw in language["corrector"].engine.words
        exact_elsewhere = any(
            normalized_raw in item["corrector"].engine.words
            for item in language_correctors
            if item["name"] != dominant_language
        )
        if exact_elsewhere and not exact_here:
            options.append([(result, 0.0)])
            continue

        correction = language["corrector"].correct(
            result.raw_word,
            ocr_candidates,
            suggestion_limit=CONTEXTUAL_SUGGESTION_LIMIT,
        )
        if exact_here:
            # Keep the existing protections for valid, confident words.
            choice = choose_contextual_suggestion(
                result.raw_word,
                result.character_predictions,
                correction.suggestions,
                previous_word=_same_line_neighbour(word_results, index, -1),
                next_word=_same_line_neighbour(word_results, index, 1),
                language_model=language_model,
            )
            if choice is None:
                options.append(
                    [(replace(result, detected_language=dominant_language), 0.0)]
                )
                continue
            suggestions = [choice]
        else:
            # Invalid words can need insertions/deletions as well as substitutions.
            suggestions = [
                suggestion
                for suggestion in correction.suggestions
                if language["corrector"]._is_reliable(suggestion, normalized_raw)
                and (
                    suggestion.edit_distance == 0
                    or word_edit_distance(
                        normalized_raw, normalize_word(suggestion.word)
                    )
                    <= language["corrector"].max_edit_distance
                )
            ]

        if not suggestions:
            options.append([(result, 0.0)])
            continue

        word_options = []
        for suggestion in suggestions:
            suggestion_words = [suggestion.word] + [
                item.word
                for item in correction.suggestions
                if item.word != suggestion.word
            ]
            revised = replace(
                result,
                final_word=suggestion.word,
                detected_language=dominant_language,
                spelling_status="possible error",
                suggested_correction=suggestion.word,
                suggestions=suggestion_words[:3],
            )
            word_options.append(
                (
                    revised,
                    language["corrector"]._ranking_score(suggestion, normalized_raw),
                )
            )
        options.append(word_options)

    # Decode together so a neighbour's revised spelling contributes immediately.
    # Each bigram is counted once and never across a line boundary.
    beams = [(0.0, [])]
    for word_options in options:
        expanded = []
        for score, chosen in beams:
            for result, word_score in word_options:
                transition = 0.0
                if chosen and language_model is not None:
                    previous = chosen[-1]
                    if (
                        previous.character_predictions
                        and result.character_predictions
                        and previous.character_predictions[0].line_index
                        == result.character_predictions[0].line_index
                    ):
                        transition = CONTEXTUAL_BIGRAM_WEIGHT * math.log1p(
                            language_model.pair_count(
                                previous.final_word, result.final_word
                            )
                        )
                expanded.append((score + word_score + transition, chosen + [result]))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = expanded[:CONTEXTUAL_BEAM_WIDTH]

    return beams[0][1]
