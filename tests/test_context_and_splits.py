"""Regression coverage for split hypotheses and corrections inside phrases."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.context_scoring import apply_phrase_context
from src.ngram_language_model import BigramLanguageModel
from src.segment_word import CharacterSegment
from src.spell_correct import ItalianSpellCorrector, OCRCandidate
from src.text_recognizer import TextRecognizer, WordRecognitionResult


class SplitHypothesisTests(unittest.TestCase):
    def choose_split(self, confidences):
        recognizer = TextRecognizer.__new__(TextRecognizer)
        recognizer.device = torch.device("cpu")
        probabilities = torch.tensor(
            [[confidence] + [(1 - confidence) / 25] * 25 for confidence in confidences]
        )
        recognizer.model = lambda batch: probabilities.log()
        whole = CharacterSegment((0, 0, 24, 20))
        parts = [CharacterSegment((0, 0, 12, 20)), CharacterSegment((12, 0, 24, 20))]
        with patch(
            "src.text_recognizer.preprocess_character",
            return_value=(torch.zeros(1, 1, 28, 28), None),
        ):
            return recognizer._choose_character_split(
                torch.zeros(20, 24, 3).numpy(), whole, parts
            )

    def test_confident_whole_letter_is_not_split_for_a_small_gain(self):
        self.assertFalse(self.choose_split([0.986, 0.99, 0.99]))

    def test_two_letters_can_beat_an_unconvincing_whole(self):
        self.assertTrue(self.choose_split([0.56, 0.76, 0.99]))


class PhraseCorrectionTests(unittest.TestCase):
    def run_phrase(self, cross_line=False, with_context=True):
        with tempfile.TemporaryDirectory() as directory:
            english = Path(directory) / "en.txt"
            italian = Path(directory) / "it.txt"
            english.write_text(
                "a 100\nbottle 100\nof 100\nwater 100\nwork 1000000\nvoter 1000000\n",
                encoding="utf-8",
            )
            italian.write_text("a 1000000\n", encoding="utf-8")
            correctors = [
                {"name": "English", "corrector": ItalianSpellCorrector(english)},
                {"name": "Italian", "corrector": ItalianSpellCorrector(italian)},
            ]
            words = ["a", "botle", "of", "wotar"]
            results = []
            for index, word in enumerate(words):
                results.append(
                    WordRecognitionResult(
                        raw_word=word,
                        final_word=word,
                        average_confidence=0.99,
                        detected_language=(
                            "Italian (also found in English)"
                            if word == "a"
                            else "English"
                        ),
                        spelling_status=(
                            "correct" if word in {"a", "of"} else "possible error"
                        ),
                        suggested_correction=None,
                        suggestions=[],
                        annotated_image_path=Path("unused.png"),
                        character_predictions=[
                            SimpleNamespace(
                                best_probability=0.99,
                                uncertain=False,
                                line_index=2 if cross_line and index == 3 else 1,
                            )
                        ],
                        uncertain_positions=[],
                    )
                )
            models = (
                {
                    "English": BigramLanguageModel(
                        {
                            ("a", "bottle"): 400,
                            ("bottle", "of"): 500,
                            ("of", "water"): 900,
                        }
                    )
                }
                if with_context
                else {}
            )
            candidates = [[OCRCandidate(word, -0.1)] for word in words]
            # "work" is close to this secondary candidate, but three edits
            # from the raw OCR. It must not evade the correction-distance limit.
            candidates[-1].append(OCRCandidate("worr", -0.2))
            return apply_phrase_context(
                results,
                candidates,
                correctors,
                models,
            )

    def test_invalid_words_can_change_length_and_use_revised_neighbours(self):
        results = self.run_phrase()
        self.assertEqual(
            " ".join(item.final_word for item in results), "a bottle of water"
        )
        self.assertEqual({item.detected_language for item in results}, {"English"})
        self.assertEqual(results[-1].raw_word, "wotar")
        self.assertEqual(results[-1].suggestions[0], "water")

    def test_context_does_not_cross_line_boundaries(self):
        plain = self.run_phrase(with_context=False)
        across_lines = self.run_phrase(cross_line=True)
        self.assertEqual(plain[-1].final_word, "voter")
        self.assertEqual(across_lines[-1].final_word, plain[-1].final_word)


if __name__ == "__main__":
    unittest.main()
