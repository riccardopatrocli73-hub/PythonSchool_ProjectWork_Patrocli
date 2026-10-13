import tempfile
from argparse import Namespace
import unittest

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.build_bigram_dictionary import tokenize as tokenize_bigram_source
from src.evaluate_words import (
    levenshtein_distance,
    parse_boolean,
    should_suppress_correction,
)
from src.finetune_local_words import collect_labeled_characters
from src.language_detection import (
    analyze_languages,
    build_language_correctors,
    select_language_analysis,
)
from src.ngram_language_model import BigramLanguageModel
from src.preprocessing import preprocess_character
from src.context_scoring import (
    apply_phrase_context,
    choose_contextual_suggestion,
    select_dominant_language,
)
from src.text_recognizer import WordRecognitionResult, WordRecognizer
from src.segment_word import (
    CharacterSegment,
    group_segments_into_text,
    group_segments_into_words,
    segment_characters,
)
from src.spell_correct import (
    CorrectionResult,
    ItalianSpellCorrector,
    OCRCandidate,
    SpellingSuggestion,
    normalize_word,
)


class PreprocessingTests(unittest.TestCase):
    def test_path_and_numpy_inputs_use_the_same_pipeline(self):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "A",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            4,
        )

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "character.png"
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(image_path)

            path_tensor, _ = preprocess_character(image_path)
            array_tensor, _ = preprocess_character(image)

        self.assertEqual(tuple(array_tensor.shape), (1, 1, 28, 28))
        self.assertTrue(torch.allclose(path_tensor, array_tensor))

    def test_invalid_numpy_shape_is_rejected(self):
        invalid_image = np.zeros((10, 10, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unsupported NumPy"):
            preprocess_character(invalid_image)


class SegmentationTests(unittest.TestCase):
    def test_linked_letters_are_split_at_the_connector(self):
        image = np.full((100, 180, 3), 255, dtype=np.uint8)
        cv2.circle(image, (55, 55), 25, (0, 0, 0), 5)
        cv2.circle(image, (120, 55), 25, (0, 0, 0), 5)
        cv2.line(image, (80, 55), (95, 55), (0, 0, 0), 2)

        _, segments = segment_characters(image)

        self.assertEqual(len(segments), 2)
        self.assertFalse(any(segment.uncertain for segment in segments))

    def test_moderately_wide_joined_letters_are_split(self):
        image = np.full((90, 120, 3), 255, dtype=np.uint8)
        cv2.line(image, (30, 25), (30, 70), (0, 0, 0), 6)
        cv2.line(image, (30, 30), (48, 35), (0, 0, 0), 5)
        cv2.ellipse(image, (75, 50), (20, 23), 0, 0, 360, (0, 0, 0), 5)
        cv2.line(image, (50, 43), (57, 43), (0, 0, 0), 2)

        _, segments = segment_characters(image)

        self.assertEqual(len(segments), 2)
        self.assertFalse(any(segment.uncertain for segment in segments))

    def test_dot_is_merged_with_its_stem(self):
        image = np.full((100, 80, 3), 255, dtype=np.uint8)
        cv2.line(image, (40, 38), (40, 82), (0, 0, 0), 7)
        cv2.circle(image, (40, 20), 4, (0, 0, 0), -1)

        _, segments = segment_characters(image)

        self.assertEqual(len(segments), 1)
        self.assertLessEqual(segments[0].box[1], 16)

    def test_multiline_threshold_preserves_short_characters(self):
        image = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.line(image, (50, 30), (50, 100), (0, 0, 0), 6)
        cv2.circle(image, (150, 82), 12, (0, 0, 0), 4)
        cv2.line(image, (50, 190), (50, 260), (0, 0, 0), 6)
        cv2.circle(image, (150, 242), 12, (0, 0, 0), 4)

        _, segments = segment_characters(image)
        layout = group_segments_into_text(segments)

        self.assertEqual(len(segments), 4)
        self.assertEqual(len(layout), 2)

    def test_character_segments_are_grouped_into_words(self):
        segments = [
            CharacterSegment((0, 0, 10, 20)),
            CharacterSegment((14, 0, 24, 20)),
            CharacterSegment((70, 0, 80, 20)),
            CharacterSegment((84, 0, 94, 20)),
        ]

        groups = group_segments_into_words(segments)

        self.assertEqual([len(group) for group in groups], [2, 2])

    def test_close_character_segments_stay_in_one_word(self):
        segments = [
            CharacterSegment((0, 0, 10, 20)),
            CharacterSegment((17, 0, 27, 20)),
            CharacterSegment((34, 0, 44, 20)),
        ]

        groups = group_segments_into_words(segments)

        self.assertEqual([len(group) for group in groups], [3])

    def test_adaptive_gap_detects_words_with_natural_spacing(self):
        segments = [
            CharacterSegment((0, 0, 10, 20)),
            CharacterSegment((14, 0, 24, 20)),
            CharacterSegment((32, 0, 42, 20)),
            CharacterSegment((46, 0, 56, 20)),
        ]

        groups = group_segments_into_words(segments)

        self.assertEqual([len(group) for group in groups], [2, 2])

    def test_segments_are_grouped_by_line_before_words(self):
        segments = [
            CharacterSegment((0, 0, 10, 20)),
            CharacterSegment((14, 0, 24, 20)),
            CharacterSegment((40, 0, 50, 20)),
            CharacterSegment((0, 50, 10, 70)),
            CharacterSegment((14, 50, 24, 70)),
            CharacterSegment((40, 50, 50, 70)),
        ]

        layout = group_segments_into_text(segments)

        self.assertEqual(
            [[len(word) for word in line] for line in layout],
            [[2, 1], [2, 1]],
        )


class SpellCorrectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dictionary_path = Path(self.temporary_directory.name) / "dictionary.txt"
        self.dictionary_path.write_text(
            "gatto 1000\n" "gatta 800\n" "casa 900\n" "za 10\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_exact_dictionary_beam_candidate_is_preferred(self):
        corrector = ItalianSpellCorrector(self.dictionary_path)
        candidates = [
            OCRCandidate("aatto", -0.1),
            OCRCandidate("gatto", -0.2),
        ]

        result = corrector.correct("aatto", candidates)

        self.assertFalse(result.is_correct)
        self.assertEqual(result.correction, "gatto")

    def test_short_ambiguous_word_is_not_forced(self):
        corrector = ItalianSpellCorrector(self.dictionary_path)
        result = corrector.correct(
            "zz",
            [OCRCandidate("zz", -0.1)],
        )

        self.assertIsNone(result.correction)

    def test_ocr_score_can_beat_higher_frequency_suggestion(self):
        self.dictionary_path.write_text(
            "favore 1000000\n" "amore 900000\n" "cuore 10\n" "muore 8\n",
            encoding="utf-8",
        )
        corrector = ItalianSpellCorrector(self.dictionary_path)
        candidates = [
            OCRCandidate("quore", -0.05),
            OCRCandidate("auore", -4.85),
            OCRCandidate("favore", -13.34),
        ]

        result = corrector.correct("quore", candidates, suggestion_limit=3)

        self.assertEqual(result.correction, "cuore")
        self.assertEqual(result.suggestions[0].word, "cuore")

    def test_character_order_can_beat_frequency_for_close_words(self):
        self.dictionary_path.write_text(
            "dato 102947\n" "gatto 11856\n" "lato 17760\n" "nato 12703\n",
            encoding="utf-8",
        )
        corrector = ItalianSpellCorrector(self.dictionary_path)

        result = corrector.correct(
            "gato",
            [OCRCandidate("gato", -0.01)],
            suggestion_limit=3,
        )

        self.assertEqual(result.correction, "gatto")
        self.assertEqual(result.suggestions[0].word, "gatto")

    def test_optional_custom_dictionary_accepts_extra_words(self):
        self.dictionary_path.write_text(
            "quaderno 1000\n" "cadevano 900\n",
            encoding="utf-8",
        )
        custom_dictionary_path = Path(self.temporary_directory.name) / "custom.txt"
        custom_dictionary_path.write_text("cuaderno\n", encoding="utf-8")
        corrector = ItalianSpellCorrector(
            self.dictionary_path,
            custom_dictionary_path=custom_dictionary_path,
        )

        result = corrector.correct(
            "cuaderno",
            [OCRCandidate("cuaderno", -0.01)],
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.correction, "cuaderno")

    def test_extra_dictionary_accepts_english_words(self):
        english_dictionary_path = Path(self.temporary_directory.name) / "english.txt"
        english_dictionary_path.write_text("home 5000\nphone 4000\n", encoding="utf-8")
        corrector = ItalianSpellCorrector(
            self.dictionary_path,
            extra_dictionary_paths=[english_dictionary_path],
        )

        result = corrector.correct(
            "home",
            [OCRCandidate("home", -0.01)],
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.correction, "home")

    def test_language_detection_prefers_exact_english_match(self):
        italian_dictionary_path = Path(self.temporary_directory.name) / "italian.txt"
        english_dictionary_path = Path(self.temporary_directory.name) / "english.txt"
        italian_dictionary_path.write_text("pensi 1000\npenali 900\n", encoding="utf-8")
        english_dictionary_path.write_text("pencil 1000\n", encoding="utf-8")
        arguments = Namespace(
            language="auto",
            dictionary=italian_dictionary_path,
            english_dictionary=english_dictionary_path,
            max_edit_distance=2,
        )

        analyses = analyze_languages(
            "pencil",
            [OCRCandidate("pencil", -0.01)],
            build_language_correctors(arguments),
        )
        selected_analysis, _ = select_language_analysis(analyses)

        self.assertEqual(selected_analysis["name"], "English")
        self.assertTrue(selected_analysis["result"].is_correct)

    def test_normalization_preserves_searchable_letters(self):
        self.assertEqual(normalize_word("Citta!"), "citta")


class BigramDictionaryBuilderTests(unittest.TestCase):
    def test_tokenizer_preserves_single_letter_words(self):
        self.assertEqual(
            tokenize_bigram_source("Vado a casa e apro."),
            ["vado", "a", "casa", "e", "apro"],
        )


class PhraseContextTests(unittest.TestCase):
    def test_dominant_language_requires_repeated_evidence(self):
        results = [
            Namespace(detected_language="Italian (also found in English)"),
            Namespace(detected_language="Italian"),
            Namespace(detected_language="English"),
        ]

        self.assertEqual(select_dominant_language(results), "Italian")
        self.assertIsNone(
            select_dominant_language(
                [
                    Namespace(detected_language="Italian"),
                    Namespace(detected_language="English"),
                ]
            )
        )

    def test_uncertain_valid_word_can_use_a_frequent_close_candidate(self):
        predictions = [
            Namespace(best_probability=0.99, uncertain=False),
            Namespace(best_probability=0.99, uncertain=False),
            Namespace(best_probability=0.99, uncertain=False),
            Namespace(best_probability=0.71, uncertain=False),
            Namespace(best_probability=0.99, uncertain=False),
        ]
        suggestions = [
            SpellingSuggestion(
                word="porno",
                edit_distance=0,
                frequency=5736,
                ocr_log_probability=-0.34,
            ),
            SpellingSuggestion(
                word="porto",
                edit_distance=1,
                frequency=28933,
                ocr_log_probability=-1.2781,
            ),
        ]

        choice = choose_contextual_suggestion(
            "porno",
            predictions,
            suggestions,
        )

        self.assertEqual(choice.word, "porto")

    def test_high_confidence_valid_word_is_preserved(self):
        predictions = [
            Namespace(best_probability=0.99, uncertain=False) for _ in range(5)
        ]
        suggestions = [
            SpellingSuggestion(
                word="porno",
                edit_distance=0,
                frequency=5736,
                ocr_log_probability=-0.01,
            ),
            SpellingSuggestion(
                word="porto",
                edit_distance=1,
                frequency=27440,
                ocr_log_probability=-0.01,
            ),
        ]

        self.assertIsNone(
            choose_contextual_suggestion(
                "porno",
                predictions,
                suggestions,
            )
        )

    def test_bigram_context_prefers_the_candidate_used_with_its_neighbour(self):
        model = BigramLanguageModel(
            {
                ("open", "door"): 100,
                ("open", "deer"): 1,
            }
        )
        predictions = [
            Namespace(best_probability=0.99, uncertain=False),
            Namespace(best_probability=0.72, uncertain=False),
            Namespace(best_probability=0.99, uncertain=False),
            Namespace(best_probability=0.99, uncertain=False),
        ]
        suggestions = [
            SpellingSuggestion(
                word="deer",
                edit_distance=0,
                frequency=10_000,
                ocr_log_probability=-0.05,
            ),
            SpellingSuggestion(
                word="door",
                edit_distance=1,
                frequency=10_000,
                ocr_log_probability=-0.40,
            ),
        ]

        choice = choose_contextual_suggestion(
            "deer",
            predictions,
            suggestions,
            previous_word="open",
            language_model=model,
        )

        self.assertEqual(choice.word, "door")

    def test_bigram_context_protects_a_valid_uncertain_word(self):
        model = BigramLanguageModel(
            {
                ("open", "door"): 100,
                ("open", "deer"): 1,
            }
        )
        predictions = [
            Namespace(best_probability=0.75, uncertain=False) for _ in range(4)
        ]
        suggestions = [
            SpellingSuggestion(
                word="door",
                edit_distance=0,
                frequency=10_000,
                ocr_log_probability=-0.40,
            ),
            SpellingSuggestion(
                word="deer",
                edit_distance=1,
                frequency=10_000,
                ocr_log_probability=-0.05,
            ),
        ]

        self.assertIsNone(
            choose_contextual_suggestion(
                "door",
                predictions,
                suggestions,
                previous_word="open",
                language_model=model,
            )
        )

    def test_short_valid_word_uses_bigram_context_when_uncertain(self):
        model = BigramLanguageModel({("va", "al"): 183})
        predictions = [
            Namespace(best_probability=0.42, uncertain=True),
            Namespace(best_probability=0.99, uncertain=False),
        ]
        suggestions = [
            SpellingSuggestion(
                word="ra",
                edit_distance=0,
                frequency=3097,
                ocr_log_probability=-0.10,
            ),
            SpellingSuggestion(
                word="la",
                edit_distance=1,
                frequency=3_934_485,
                ocr_log_probability=-0.10,
            ),
            SpellingSuggestion(
                word="va",
                edit_distance=1,
                frequency=508_747,
                ocr_log_probability=-0.10,
            ),
        ]

        choice = choose_contextual_suggestion(
            "ra",
            predictions,
            suggestions,
            next_word="al",
            language_model=model,
        )

        self.assertEqual(choice.word, "va")

    def test_phrase_context_corrects_short_word_with_local_bigram_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            dictionary_path = Path(directory) / "dictionary.txt"
            dictionary_path.write_text(
                "ra 3097\n"
                "era 657748\n"
                "ora 559590\n"
                "tra 165237\n"
                "fra 33404\n"
                "ray 15221\n"
                "la 3934485\n"
                "ira 2957\n"
                "a 3455859\n"
                "ha 1662783\n"
                "ma 1650973\n"
                "rae 10000\n"
                "da 1323764\n"
                "raj 9000\n"
                "rap 8000\n"
                "ran 7000\n"
                "ram 6000\n"
                "va 508747\n"
                "al 745025\n"
                "barca 16976\n",
                encoding="utf-8",
            )
            corrector = ItalianSpellCorrector(dictionary_path)

            def word_result(word, probabilities):
                predictions = [
                    Namespace(
                        best_probability=probability,
                        uncertain=probability < 0.50,
                        line_index=1,
                    )
                    for probability in probabilities
                ]

                return WordRecognitionResult(
                    raw_word=word,
                    final_word=word,
                    average_confidence=sum(probabilities) / len(probabilities),
                    detected_language="Italian",
                    spelling_status="correct",
                    suggested_correction=None,
                    suggestions=[word],
                    annotated_image_path=Path(directory) / "annotated.png",
                    character_predictions=predictions,
                    uncertain_positions=[],
                )

            contextual_results = apply_phrase_context(
                [
                    word_result("barca", [0.99, 0.99, 0.99, 0.99, 0.99]),
                    word_result("ra", [0.42, 0.99]),
                    word_result("al", [0.99, 0.99]),
                ],
                [
                    [OCRCandidate("barca", -0.01)],
                    [OCRCandidate("ra", -0.10)],
                    [OCRCandidate("al", -0.01)],
                ],
                [{"name": "Italian", "corrector": corrector}],
                {"Italian": BigramLanguageModel({("va", "al"): 183})},
            )

        self.assertEqual(contextual_results[1].final_word, "va")
        self.assertEqual(contextual_results[1].suggested_correction, "va")
        self.assertEqual(contextual_results[1].suggestions[0], "va")


class QtAppPipelineTests(unittest.TestCase):
    def test_word_recognizer_returns_structured_result(self):
        image_path = Path("images/PenciL.png")
        checkpoint_path = Path("models/best_ch_cnn_local_words.pt")

        if not image_path.exists() or not checkpoint_path.exists():
            self.skipTest("Local OCR image/checkpoint not available")

        recognizer = WordRecognizer(checkpoint_path=checkpoint_path)
        result = recognizer.recognize(image_path)

        self.assertEqual(normalize_word(result.raw_word), "pencil")
        self.assertEqual(result.detected_language, "English")
        self.assertEqual(result.spelling_status, "correct")
        self.assertEqual(result.final_word, "pencil")
        self.assertEqual(len(result.character_predictions), 6)
        self.assertTrue(result.annotated_image_path.exists())

    def test_text_recognizer_handles_single_word_as_text(self):
        image_path = Path("images/PenciL.png")
        checkpoint_path = Path("models/best_ch_cnn_local_words.pt")

        if not image_path.exists() or not checkpoint_path.exists():
            self.skipTest("Local OCR image/checkpoint not available")

        recognizer = WordRecognizer(checkpoint_path=checkpoint_path)
        result = recognizer.recognize_text(image_path)

        self.assertEqual(result.raw_text, "pencil")
        self.assertEqual(result.final_text, "pencil")
        self.assertEqual(result.detected_language, "English")
        self.assertEqual(result.spelling_status, "correct")
        self.assertEqual(len(result.word_results), 1)
        self.assertTrue(result.annotated_image_path.exists())

    def test_word_recognizer_suggests_italian_word_for_gato(self):
        image_path = Path("images/Gato.png")
        checkpoint_path = Path("models/best_ch_cnn_local_words.pt")

        if not image_path.exists() or not checkpoint_path.exists():
            self.skipTest("Local OCR image/checkpoint not available")

        recognizer = WordRecognizer(checkpoint_path=checkpoint_path)
        result = recognizer.recognize(image_path)

        self.assertEqual(normalize_word(result.raw_word), "gato")
        self.assertEqual(result.detected_language, "Italian")
        self.assertEqual(result.spelling_status, "possible error")
        self.assertEqual(result.suggested_correction, "gatto")
        self.assertEqual(result.final_word, "gatto")
        self.assertEqual(result.suggestions[0], "gatto")

    def test_word_recognizer_suggests_italian_dictionary_word(self):
        image_path = Path("images/cuaderno.png")
        checkpoint_path = Path("models/best_ch_cnn_local_words.pt")

        if not image_path.exists() or not checkpoint_path.exists():
            self.skipTest("Local OCR image/checkpoint not available")

        recognizer = WordRecognizer(checkpoint_path=checkpoint_path)
        result = recognizer.recognize(image_path)

        self.assertEqual(normalize_word(result.raw_word), "cuaderno")
        self.assertEqual(result.detected_language, "Italian")
        self.assertEqual(result.spelling_status, "possible error")
        self.assertEqual(result.suggested_correction, "quaderno")
        self.assertEqual(result.final_word, "quaderno")
        self.assertEqual(result.suggestions[0], "quaderno")


class EvaluationHelperTests(unittest.TestCase):
    def test_levenshtein_distance_handles_insertions(self):
        self.assertEqual(levenshtein_distance("gato", "gatto"), 1)

    def test_boolean_parser_accepts_manifest_values(self):
        self.assertTrue(parse_boolean("si"))
        self.assertFalse(parse_boolean("false"))

        with self.assertRaises(ValueError):
            parse_boolean("forse")

    def test_conservative_mode_keeps_high_confidence_raw_ocr(self):
        correction = CorrectionResult(
            raw_word="quore",
            is_correct=False,
            correction="favore",
            suggestions=[],
        )

        self.assertTrue(
            should_suppress_correction(
                correction=correction,
                raw_word="quore",
                average_confidence=0.99,
                segmentation_uncertain=False,
                correction_mode="conservative",
                preserve_raw_threshold=0.95,
            )
        )
        self.assertFalse(
            should_suppress_correction(
                correction=correction,
                raw_word="quore",
                average_confidence=0.80,
                segmentation_uncertain=False,
                correction_mode="conservative",
                preserve_raw_threshold=0.95,
            )
        )


class LocalFineTuningTests(unittest.TestCase):
    def test_manifest_text_is_used_instead_of_image_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            image = np.full((80, 80, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                "A",
                (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                4,
            )
            cv2.imwrite(str(directory_path / "sample_001.png"), image)

            manifest_path = directory_path / "labels.csv"
            manifest_path.write_text(
                "image,expected_text\n" "sample_001.png,a\n",
                encoding="utf-8",
            )

            samples, skipped = collect_labeled_characters(
                images_dir=directory_path,
                manifest_path=manifest_path,
                include_single_characters=True,
            )

        self.assertFalse(skipped)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].label, 0)
        self.assertEqual(samples[0].source, "sample_001.png")


if __name__ == "__main__":
    unittest.main()
