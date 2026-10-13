""" Phrase-level language selection and conservative contextual scoring. """

import math
from dataclasses import replace

from src.spell_correct import normalize_word

# Costants used to avoid aggressive corrections
CONTEXTUAL_CONFIDENCE_THRESHOLD = 0.85
CONTEXTUAL_EDIT_PENALTY = 0.25
CONTEXTUAL_MINIMUM_GAIN = 0.35
CONTEXTUAL_BIGRAM_WEIGHT = 1.25
CONTEXTUAL_FREQUENCY_WEIGHT = 0.20
CONTEXTUAL_SUGGESTION_LIMIT = 25
CONTEXTUAL_SHORT_WORD_LENGTH = 4

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
    """ Choose a safer alternative for an uncertain valid dictionary word. """

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
    """ Apply language-consistent corrections to uncertain valid words. """

    if len(word_results) < 2:
        return word_results

    dominant_language = select_dominant_language(word_results)

    if dominant_language is None:
        return word_results

    # Find corrector
    language = next(
        (item for item in language_correctors if item["name"] == dominant_language),
        None,
    )

    if language is None:
        return word_results

    contextual_results = []
    language_model = (bigram_models or {}).get(dominant_language)

    for index, (result, ocr_candidates) in enumerate(
        zip(word_results, ocr_candidate_groups)
    ):
        if result.spelling_status != "correct":
            contextual_results.append(result)
            continue

        correction = language["corrector"].correct(
            result.raw_word,
            ocr_candidates,
            suggestion_limit=CONTEXTUAL_SUGGESTION_LIMIT,
        )
        contextual_suggestion = choose_contextual_suggestion(
            result.raw_word,
            result.character_predictions,
            correction.suggestions,
            previous_word=_same_line_neighbour(word_results, index, -1),
            next_word=_same_line_neighbour(word_results, index, 1),
            language_model=language_model,
        )

        if contextual_suggestion is None:
            contextual_results.append(result)
            continue

        suggestion_words = [contextual_suggestion.word]
        suggestion_words.extend(
            suggestion.word
            for suggestion in correction.suggestions
            if suggestion.word != contextual_suggestion.word
        )
        contextual_results.append(
            replace(
                result,
                final_word=contextual_suggestion.word,
                detected_language=dominant_language,
                spelling_status="possible error",
                suggested_correction=contextual_suggestion.word,
                suggestions=suggestion_words[:3],
            )
        )

    return contextual_results
