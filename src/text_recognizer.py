"""Reusable handwritten word and text recognition pipeline."""

""" 
This module builds the OCR pipeline. 
- INPUT: Image
- OPERATIONS: Image division in words/characters, text correction with context
              and dictionaries
- OUTPUT: Result for GUI or CLI
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch
import torch.nn.functional as F

from src.context_scoring import apply_phrase_context
from src.alphabet import LETTERS
from src.language_detection import (
    DEFAULT_DICTIONARY,
    DEFAULT_ENGLISH_DICTIONARY,
    analyze_languages,
    build_language_correctors,
    format_language_name,
    select_language_analysis,
)
from src.ngram_language_model import load_bigram_models
from src.predict_character import load_model
from src.preprocessing import preprocess_character
from src.segment_word import group_segments_into_text, segment_characters
from src.spell_correct import build_ocr_candidates

# Main Paths definition
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qt_predictions"
LOCAL_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn_local_words.pt"
# Use model best_ch_cnn_local_words.pt if exists
DEFAULT_QT_CHECKPOINT = (
    LOCAL_CHECKPOINT if LOCAL_CHECKPOINT.exists() else DEFAULT_CHECKPOINT
)

"""
------------------------------------------------------
This class represents the result of a single character
For each ch stores:
- Position
- Coordinates in the image
- Best letter
- Probability
- Alternatives if uncertain on the prediction
------------------------------------------------------
"""
@dataclass(frozen=True)
class CharacterPrediction:
    position: int
    box: tuple[int, int, int, int]
    best_letter: str
    best_probability: float
    alternatives: list[tuple[str, float]]
    uncertain: bool
    line_index: int = 1
    word_index: int = 1
    word_position: int = 1

"""
------------------------------------------------------
This class represents the recognized word
- raw_word: word read by the model
- final_word: word after revision
------------------------------------------------------
"""
@dataclass(frozen=True)
class WordRecognitionResult:
    raw_word: str
    final_word: str
    average_confidence: float
    detected_language: str
    spelling_status: str
    suggested_correction: str | None
    suggestions: list[str]
    annotated_image_path: Path
    character_predictions: list[CharacterPrediction]
    uncertain_positions: list[int]

"""
------------------------------------------------------
This class represents the whole text
------------------------------------------------------
"""
@dataclass(frozen=True)
class TextRecognitionResult:
    raw_text: str
    final_text: str
    average_confidence: float
    detected_language: str
    spelling_status: str
    suggested_correction: str | None
    suggestions: list[str]
    annotated_image_path: Path
    character_predictions: list[CharacterPrediction]
    uncertain_positions: list[int]
    word_results: list[WordRecognitionResult]
    line_count: int

"""
------------------------------------------------------
This class represents the text_recognizer
------------------------------------------------------
"""
class TextRecognizer:
    def __init__(
        self,
        checkpoint_path=DEFAULT_QT_CHECKPOINT,
        italian_dictionary=DEFAULT_DICTIONARY,
        english_dictionary=DEFAULT_ENGLISH_DICTIONARY,
        top_k=3,
        beam_width=50,
        confidence_threshold=0.50,
        max_edit_distance=2,
        output_dir=DEFAULT_OUTPUT_DIR,
        bigram_dictionaries=None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.italian_dictionary = Path(italian_dictionary)
        self.english_dictionary = Path(english_dictionary)
        self.top_k = top_k
        self.beam_width = beam_width
        self.confidence_threshold = confidence_threshold
        self.max_edit_distance = max_edit_distance
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.checkpoint_path, self.device)
        self.language_correctors = build_language_correctors(self._language_arguments())
        self.bigram_models = load_bigram_models(bigram_dictionaries)

    # Language object definition
    def _language_arguments(self):
        return SimpleNamespace(
            language="auto",
            dictionary=self.italian_dictionary,
            english_dictionary=self.english_dictionary,
            max_edit_distance=self.max_edit_distance,
        )

    # Annotated image creation
    # Visual Output: bigger image, padding addition, rectangles drawing,
    # uncertain ch different color, ch indexing
    def _make_display_image(self, image, predictions):
        image_height, image_width = image.shape[:2]
        target_width = 1600
        target_height = 900
        scale = min(
            6.0,
            max(
                1.0,
                min(
                    target_width / max(image_width, 1),
                    target_height / max(image_height, 1),
                ),
            ),
        )
        scaled_width = max(1, round(image_width * scale))
        scaled_height = max(1, round(image_height * scale))
        top_padding = 72
        bottom_padding = 48
        side_padding = 32

        enlarged = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_CUBIC,
        )
        canvas = cv2.copyMakeBorder(
            enlarged,
            top_padding,
            bottom_padding,
            side_padding,
            side_padding,
            cv2.BORDER_CONSTANT,
            value=(248, 249, 250),
        )

        for prediction in predictions:
            left, top, right, bottom = prediction.box
            x1 = round(left * scale) + side_padding
            y1 = round(top * scale) + top_padding
            x2 = round(right * scale) + side_padding
            y2 = round(bottom * scale) + top_padding
            color = (0, 150, 255) if prediction.uncertain else (40, 150, 80)
            label_color = (0, 125, 215) if prediction.uncertain else (25, 120, 70)
            thickness = max(2, round(scale / 2))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=thickness)
            label = str(prediction.position)
            font_scale = 0.55
            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                2,
            )
            label_radius = max(15, (text_size[0] + 10) // 2)
            label_center = (
                x1 + label_radius,
                max(label_radius + 4, y1 - label_radius - 6),
            )
            cv2.circle(
                canvas,
                label_center,
                label_radius,
                label_color,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                label,
                (
                    label_center[0] - text_size[0] // 2,
                    label_center[1] + text_size[1] // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return canvas

    # Characters prediction
    def _predict_segments(
        self,
        image,
        segments, # List of detected ch
        start_position=1,
        line_index=1,
        word_index=1,
    ):
        if not segments:
            raise RuntimeError("No characters detected")

        character_tensors = []

        # coordinates
        for segment in segments:
            left, top, right, bottom = segment.box
            character = image[top:bottom, left:right]
            # ch preprocessing
            tensor, _ = preprocess_character(character, trim_connectors=True)
            character_tensors.append(tensor)

        # tensors joint in a batch
        character_batch = torch.cat(character_tensors, dim=0).to(self.device)

        # Model inference
        with torch.inference_mode():
            logits = self.model(character_batch)
            probabilities = F.softmax(logits, dim=1)
            # takes best top_k letters for each ch
            top_probabilities, top_labels = torch.topk(
                probabilities,
                k=self.top_k,
                dim=1,
            )

        top_probabilities = top_probabilities.cpu()
        top_labels = top_labels.cpu()
        # raw word building
        raw_word = "".join(LETTERS[int(label)] for label in top_labels[:, 0])
        # average confidence computation
        average_confidence = top_probabilities[:, 0].mean().item()
        # alternative candidates
        ocr_candidates = build_ocr_candidates(
            top_probabilities,
            top_labels,
            beam_width=self.beam_width,
        )

        # creation of a lif of uncertain ch
        predictions = []
        uncertain_positions = []

        for offset, (segment, probability_values, label_values) in enumerate(
            zip(segments, top_probabilities, top_labels),
            start=0,
        ):
            position = start_position + offset
            word_position = offset + 1
            best_label = label_values[0].item()
            best_probability = probability_values[0].item()
            best_letter = LETTERS[best_label]
            uncertain = (
                segment.uncertain or best_probability < self.confidence_threshold
            )

            if uncertain:
                uncertain_positions.append(position)

            alternatives = [
                (LETTERS[int(label.item())], float(probability.item()))
                for probability, label in zip(probability_values, label_values)
            ]
            predictions.append(
                CharacterPrediction(
                    position=position,
                    box=segment.box,
                    best_letter=best_letter,
                    best_probability=best_probability,
                    alternatives=alternatives,
                    uncertain=uncertain,
                    line_index=line_index,
                    word_index=word_index,
                    word_position=word_position,
                )
            )

        return (
            raw_word,
            average_confidence,
            ocr_candidates,
            predictions,
            uncertain_positions,
        )

    # Takes a raw word and OCR candidates, analyzes words in different languages,
    # chooses best analysis, decides ortographic state and fills if a correction is found
    def _correct_word(self, raw_word, ocr_candidates):
        detected_language = "unknown"
        spelling_status = "unknown word"
        suggested_correction = None
        suggestions = []

        analyses = analyze_languages(
            raw_word,
            ocr_candidates,
            self.language_correctors,
        )
        selected_analysis, exact_matches = select_language_analysis(analyses)

        detected_language = format_language_name(selected_analysis, exact_matches)

        if selected_analysis is not None:
            correction = selected_analysis["result"]
            suggestions = [suggestion.word for suggestion in correction.suggestions]

            if correction.is_correct:
                spelling_status = "correct"
            elif correction.correction:
                spelling_status = "possible error"
                suggested_correction = correction.correction
            else:
                spelling_status = "unknown word"

        return detected_language, spelling_status, suggested_correction, suggestions

    # Returns final decision
    def _build_word_result(
        self,
        raw_word,
        average_confidence,
        ocr_candidates,
        predictions,
        uncertain_positions,
        annotated_image_path,
    ):
        (
            detected_language,
            spelling_status,
            suggested_correction,
            suggestions,
        ) = self._correct_word(raw_word, ocr_candidates)
        final_word = suggested_correction or raw_word

        return WordRecognitionResult(
            raw_word=raw_word,
            final_word=final_word,
            average_confidence=average_confidence,
            detected_language=detected_language,
            spelling_status=spelling_status,
            suggested_correction=suggested_correction,
            suggestions=suggestions,
            annotated_image_path=annotated_image_path,
            character_predictions=predictions,
            uncertain_positions=uncertain_positions,
        )

    # Takes single words result and produces a language for the text
    def _combine_languages(self, word_results):
        languages = []

        for result in word_results:
            if result.detected_language == "unknown":
                continue

            for language in result.detected_language.split(" (also found in "):
                cleaned = language.rstrip(")")
                if cleaned and cleaned not in languages:
                    languages.append(cleaned)

        if not languages:
            return "unknown"

        if len(languages) == 1:
            return languages[0]

        return " + ".join(languages)

    # Return the status of the output
    def _combine_status(self, word_results):
        statuses = [result.spelling_status for result in word_results]

        if any(status == "possible error" for status in statuses):
            return "possible error" if len(statuses) == 1 else "possible errors"

        if statuses and all(status == "correct" for status in statuses):
            return "correct"

        if any(status == "unknown word" for status in statuses):
            return "unknown words"

        return "not checked"

    def recognize(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        binary, segments = segment_characters(image)
        output_dir = self.output_dir / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_image_path = output_dir / "word_prediction.png"

        (
            raw_word,
            average_confidence,
            ocr_candidates,
            predictions,
            uncertain_positions,
        ) = self._predict_segments(image, segments)
        result = self._build_word_result(
            raw_word,
            average_confidence,
            ocr_candidates,
            predictions,
            uncertain_positions,
            annotated_image_path,
        )

        display_image = self._make_display_image(image, result.character_predictions)
        cv2.imwrite(str(output_dir / "binary.png"), binary)
        cv2.imwrite(str(annotated_image_path), display_image)

        return result

    # Main function for GUI
    def recognize_text(self, image_path):

        # Path check and image reading
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # ch segmentation
        binary, segments = segment_characters(image)
        # group ch in lines and words. Every word is a list of segments
        text_layout = group_segments_into_text(segments)
        word_groups = [word for line in text_layout for word in line]

        if not word_groups:
            raise RuntimeError("No words detected")

        output_dir = self.output_dir / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_image_path = output_dir / "text_prediction.png"

        word_results = []
        word_ocr_candidates = []
        character_predictions = []
        uncertain_positions = []
        line_word_results = []
        next_position = 1
        next_word_index = 1

        # for every line and word, call _predict_segments and create a WordRecognitionResuly
        for line_index, line_words in enumerate(text_layout, start=1):
            current_line_results = []

            for word_segments in line_words:
                (
                    raw_word,
                    average_confidence,
                    ocr_candidates,
                    predictions,
                    word_uncertain_positions,
                ) = self._predict_segments(
                    image,
                    word_segments,
                    start_position=next_position,
                    line_index=line_index,
                    word_index=next_word_index,
                )
                word_result = self._build_word_result(
                    raw_word,
                    average_confidence,
                    ocr_candidates,
                    predictions,
                    word_uncertain_positions,
                    annotated_image_path,
                )
                current_line_results.append(word_result)
                word_results.append(word_result)
                word_ocr_candidates.append(ocr_candidates)
                character_predictions.extend(predictions)
                uncertain_positions.extend(word_uncertain_positions)
                next_position += len(predictions)
                next_word_index += 1

            line_word_results.append(current_line_results)

        # context application
        word_results = apply_phrase_context(
            word_results,
            word_ocr_candidates,
            self.language_correctors,
            self.bigram_models,
        )

        # reconstruct words and lines after context analysis
        contextual_result_iterator = iter(word_results)
        line_word_results = [
            [next(contextual_result_iterator) for _ in line_words]
            for line_words in text_layout
        ]

        raw_text = "\n".join(
            " ".join(result.raw_word for result in line_results)
            for line_results in line_word_results
        )
        # final text construction
        final_text = "\n".join(
            " ".join(result.final_word for result in line_results)
            for line_results in line_word_results
        )
        total_characters = sum(
            len(result.character_predictions) for result in word_results
        )
        # mean weighted confidence computation
        weighted_confidence = sum(
            result.average_confidence * len(result.character_predictions)
            for result in word_results
        )
        average_confidence = weighted_confidence / max(total_characters, 1)
        # language and spelling status combination
        detected_language = self._combine_languages(word_results)
        spelling_status = self._combine_status(word_results)
        suggested_correction = final_text if final_text != raw_text else None
        suggestions = [
            f"{result.raw_word} -> {result.suggested_correction}"
            for result in word_results
            if result.suggested_correction
        ]

        display_image = self._make_display_image(image, character_predictions)
        cv2.imwrite(str(output_dir / "binary.png"), binary)
        cv2.imwrite(str(annotated_image_path), display_image)

        return TextRecognitionResult(
            raw_text=raw_text,
            final_text=final_text,
            average_confidence=average_confidence,
            detected_language=detected_language,
            spelling_status=spelling_status,
            suggested_correction=suggested_correction,
            suggestions=suggestions,
            annotated_image_path=annotated_image_path,
            character_predictions=character_predictions,
            uncertain_positions=uncertain_positions,
            word_results=word_results,
            line_count=len(text_layout),
        )


# Backward-compatible name for existing callers.
WordRecognizer = TextRecognizer
