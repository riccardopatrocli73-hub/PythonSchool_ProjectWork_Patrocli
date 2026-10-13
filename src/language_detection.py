"""Bilingual dictionary loading, language detection, and result selection."""

from pathlib import Path

from src.spell_correct import ItalianSpellCorrector, normalize_word

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICTIONARY = PROJECT_ROOT / "data" / "dictionaries" / "it_frequency.txt"
DEFAULT_ENGLISH_DICTIONARY = PROJECT_ROOT / "data" / "dictionaries" / "en_frequency.txt"

# This function creates a corrector by using a dictionary
def build_spell_corrector(dictionary_path, max_edit_distance):
    if dictionary_path is None or not dictionary_path.exists():
        return None

    return ItalianSpellCorrector(
        dictionary_path,
        max_edit_distance=max_edit_distance,
    )

# This function builds the list of correctors
def build_language_correctors(arguments):
    requested_languages = []

    if arguments.language in {"auto", "it"}:
        requested_languages.append(
            (
                "it",
                "Italian",
                arguments.dictionary,
            )
        )

    if arguments.language in {"auto", "en"}:
        requested_languages.append(
            (
                "en",
                "English",
                arguments.english_dictionary,
            )
        )

    correctors = []

    for code, name, dictionary_path in requested_languages:
        corrector = build_spell_corrector(
            dictionary_path,
            max_edit_distance=arguments.max_edit_distance,
        )

        if corrector is None:
            continue

        correctors.append(
            {
                "code": code,
                "name": name,
                "dictionary": dictionary_path,
                "corrector": corrector,
            }
        )

    return correctors

# This function assigns a score to a language based on corrector result
def language_score(analysis):
    result = analysis["result"]
    corrector = analysis["corrector"]

    if result.is_correct:
        normalized = normalize_word(result.raw_word)
        frequency = corrector.surface_frequencies.get(normalized, 0)
        return (3, frequency)  # 3 indicates max priority, frequency is used to choose the language

    if result.suggestions:
        best_suggestion = result.suggestions[0]
        return (
            2,
            corrector._ranking_score(
                best_suggestion,
                normalize_word(result.raw_word),
            ),
        )

    return (0, float("-inf"))  # If no word is found

# This fuunction tries a OCR word against all possible correctors
def analyze_languages(raw_word, ocr_candidates, language_correctors):
    analyses = []

    for language in language_correctors:
        result = language["corrector"].correct(
            raw_word,
            ocr_candidates,
            suggestion_limit=3,
        )
        # Each analysis contains a language, dictionary,
        # correction result and language score
        analyses.append(
            {
                **language,
                "result": result,
                "score": language_score(
                    {
                        **language,
                        "result": result,
                    }
                ),
            }
        )

    return analyses

# This function chooses the best analysis
def select_language_analysis(analyses):
    if not analyses:
        return None, []

    # Exact matches searching
    exact_matches = [analysis for analysis in analyses if analysis["result"].is_correct]

    if exact_matches:
        exact_matches.sort(
            key=lambda analysis: analysis["score"],
            reverse=True,
        )
        return exact_matches[0], exact_matches

    analyses.sort(
        key=lambda analysis: analysis["score"],
        reverse=True,
    )

    return analyses[0], []

# This function produces name of the language
def format_language_name(selected_analysis, exact_matches):
    if not selected_analysis:
        return "unknown"

    if len(exact_matches) > 1:
        other_names = [
            analysis["name"]
            for analysis in exact_matches
            if analysis["name"] != selected_analysis["name"]
        ]
        return (
            selected_analysis["name"]
            + " (also found in "
            + ", ".join(other_names)
            + ")"
        )

    return selected_analysis["name"]
