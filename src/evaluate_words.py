""" Pipeline Evaluation """

import argparse
import csv

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from src.alphabet import LETTERS
from src.predict_character import load_model
from src.preprocessing import preprocess_character
from src.segment_word import segment_characters
from src.spell_correct import (
    ItalianSpellCorrector,
    build_ocr_candidates,
    normalize_word,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn_local_words.pt"
DEFAULT_DICTIONARY = PROJECT_ROOT / "data" / "dictionaries" / "it_frequency.txt"
DEFAULT_ENGLISH_DICTIONARY = PROJECT_ROOT / "data" / "dictionaries" / "en_frequency.txt"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "word_evaluation.csv"


def resolve_extra_dictionaries(extra_dictionaries):
    dictionaries = []

    if DEFAULT_ENGLISH_DICTIONARY.exists():
        dictionaries.append(DEFAULT_ENGLISH_DICTIONARY)

    dictionaries.extend(extra_dictionaries or [])

    return dictionaries


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the complete word recognition pipeline"
    )

    parser.add_argument(
        "manifest",
        type=Path,
        help="CSV file containing the evaluation samples",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing the word images",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained CNN checkpoint",
    )

    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_DICTIONARY,
        help="Path to the Italian frequency dictionary",
    )

    parser.add_argument(
        "--custom-dictionary",
        type=Path,
        default=None,
        help="Optional extra words accepted by the spell corrector.",
    )

    parser.add_argument(
        "--extra-dictionary",
        type=Path,
        action="append",
        default=None,
        help="Additional weighted frequency dictionary. Can be passed more than once.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path of the detailed evaluation CSV",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Character predictions used by beam search",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=50,
        help="Maximum number of OCR candidates retained",
    )

    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance used by the corrector",
    )

    parser.add_argument(
        "--language",
        choices=("it", "none"),
        default="it",
        help="Language corrector to run after OCR.",
    )

    parser.add_argument(
        "--correction-mode",
        choices=("aggressive", "conservative"),
        default="conservative",
        help=(
            "Use conservative to keep a high-confidence raw OCR word instead "
            "of replacing it with a low-context dictionary suggestion."
        ),
    )

    parser.add_argument(
        "--preserve-raw-threshold",
        type=float,
        default=0.95,
        help="Average OCR confidence used by --correction-mode conservative.",
    )

    return parser.parse_args()


def parse_boolean(value):
    normalized_value = str(value).strip().lower()

    if normalized_value in {"true", "1", "yes", "y", "si", "s"}:
        return True

    if normalized_value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Invalid boolean value: {value}. " "Use true/false, yes/no or 1/0."
    )


def parse_optional_boolean(value, default):
    if value is None or str(value).strip() == "":
        return default

    return parse_boolean(value)


def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def levenshtein_distance(first_word, second_word):
    if first_word == second_word:
        return 0

    if not first_word:
        return len(second_word)

    if not second_word:
        return len(first_word)

    previous_row = list(range(len(second_word) + 1))

    for first_index, first_character in enumerate(
        first_word,
        start=1,
    ):
        current_row = [first_index]

        for second_index, second_character in enumerate(
            second_word,
            start=1,
        ):
            insertion_cost = current_row[second_index - 1] + 1
            deletion_cost = previous_row[second_index] + 1
            substitution_cost = previous_row[second_index - 1]

            if first_character != second_character:
                substitution_cost += 1

            current_row.append(
                min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def resolve_image_path(image_value, images_directory):
    image_path = Path(image_value)

    if image_path.is_absolute():
        return image_path

    return images_directory / image_path


def load_manifest(manifest_path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        required_columns = {
            "image",
            "expected_text",
            "is_correct",
            "expected_correction",
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))

            raise ValueError(f"Missing manifest columns: {missing_text}")

        samples = list(reader)

    if not samples:
        raise ValueError(f"The manifest contains no samples: {manifest_path}")

    return samples


def predict_word(
    image_path,
    model,
    device,
    corrector,
    top_k,
    beam_width,
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    # segmentation
    _, segments = segment_characters(image)

    if not segments:
        raise RuntimeError(f"No characters found in image: {image_path}")

    character_tensors = []

    # preprocessing
    for position, segment in enumerate(segments):
        left, top, right, bottom = segment.box
        character = image[top:bottom, left:right]

        if character.size == 0:
            raise RuntimeError(f"Empty crop at character position {position + 1}")

        tensor, _ = preprocess_character(
            character,
            trim_connectors=True,
        )
        character_tensors.append(tensor)

    character_batch = torch.cat(
        character_tensors,
        dim=0,
    ).to(device)

    with torch.inference_mode():
        logits = model(character_batch)
        probabilities = F.softmax(logits, dim=1)

        top_probabilities, top_labels = torch.topk(
            probabilities,
            k=top_k,
            dim=1,
        )

    top_probabilities = top_probabilities.cpu()
    top_labels = top_labels.cpu()

    raw_word = "".join(LETTERS[int(label)] for label in top_labels[:, 0])

    ocr_candidates = build_ocr_candidates(
        top_probabilities,
        top_labels,
        beam_width=beam_width,
    )

    correction = None

    if corrector is not None:
        correction = corrector.correct(
            raw_word,
            ocr_candidates,
            suggestion_limit=3,
        )

    return {
        "segment_count": len(segments),
        "segmentation_uncertain": any(segment.uncertain for segment in segments),
        "raw_word": raw_word,
        "average_confidence": top_probabilities[:, 0].mean().item(),
        "ocr_candidates": ocr_candidates,
        "correction": correction,
    }

# Maintain OCR word if confidence above threshold
def should_suppress_correction(
    correction,
    raw_word,
    average_confidence,
    segmentation_uncertain,
    correction_mode,
    preserve_raw_threshold,
):
    if correction is None or correction.correction is None:
        return False

    return (
        correction_mode == "conservative"
        and normalize_word(correction.correction) != normalize_word(raw_word)
        and average_confidence >= preserve_raw_threshold
        and not segmentation_uncertain
    )

# Decision between raw word or correction
def choose_final_text(prediction, correction_suppressed):
    correction = prediction["correction"]

    if correction is not None and correction.correction and not correction_suppressed:
        return correction.correction

    return prediction["raw_word"]

# Comparison between result and expectation based on Levenshtein distance
def evaluate_prediction(
    prediction,
    expected_text,
    expected_is_correct,
    expected_correction,
    expected_final,
    evaluate_spelling,
    correction_suppressed,
):
    normalized_expected = normalize_word(expected_text)
    normalized_raw = normalize_word(prediction["raw_word"])
    normalized_expected_final = normalize_word(expected_final)

    correction = prediction["correction"]
    final_text = choose_final_text(prediction, correction_suppressed)
    normalized_final = normalize_word(final_text)

    segmentation_ok = prediction["segment_count"] == len(normalized_expected)

    character_distance = levenshtein_distance(
        normalized_raw,
        normalized_expected,
    )

    raw_word_correct = normalized_raw == normalized_expected
    final_text_correct = normalized_final == normalized_expected_final

    predicted_is_correct = None
    spelling_detection_correct = None
    correction_top_1 = None
    correction_top_3 = None

    if evaluate_spelling:
        if correction is None:
            predicted_is_correct = True
            normalized_correction = ""
            normalized_suggestions = []
        else:
            predicted_is_correct = correction.is_correct
            normalized_correction = normalize_word(correction.correction or "")
            normalized_suggestions = [
                normalize_word(suggestion.word)
                for suggestion in correction.suggestions[:3]
            ]

        normalized_expected_correction = normalize_word(expected_correction)
        spelling_detection_correct = predicted_is_correct == expected_is_correct

        correction_top_1 = False
        correction_top_3 = False

        if not expected_is_correct:
            correction_top_1 = normalized_correction == normalized_expected_correction
            correction_top_3 = normalized_expected_correction in normalized_suggestions

    if not segmentation_ok:
        error_type = "SEGMENTATION_ERROR"
    elif not raw_word_correct:
        error_type = "OCR_ERROR"
    elif not final_text_correct:
        error_type = "FINAL_TEXT_ERROR"
    elif evaluate_spelling and not spelling_detection_correct:
        error_type = "SPELLING_DETECTION_ERROR"
    elif evaluate_spelling and not expected_is_correct and not correction_top_1:
        error_type = "SPELLING_CORRECTION_ERROR"
    else:
        error_type = "CORRECT"

    return {
        "segmentation_ok": segmentation_ok,
        "character_distance": character_distance,
        "raw_word_correct": raw_word_correct,
        "final_text": final_text,
        "final_text_correct": final_text_correct,
        "predicted_is_correct": predicted_is_correct,
        "spelling_detection_correct": spelling_detection_correct,
        "correction_top_1": correction_top_1,
        "correction_top_3": correction_top_3,
        "error_type": error_type,
    }


def format_candidates(candidates, limit=5):
    return " | ".join(candidate.word for candidate in candidates[:limit])


def format_suggestions(correction, limit=3):
    if correction is None:
        return ""

    return " | ".join(suggestion.word for suggestion in correction.suggestions[:limit])


def write_results(output_path, results):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "image",
        "category",
        "expected_text",
        "expected_final",
        "evaluate_spelling",
        "expected_is_correct",
        "expected_correction",
        "segment_count",
        "expected_length",
        "segmentation_ok",
        "segmentation_uncertain",
        "raw_ocr",
        "raw_word_correct",
        "ocr_candidates",
        "average_confidence",
        "predicted_is_correct",
        "spelling_detection_correct",
        "suggested_correction",
        "correction_suppressed",
        "final_text",
        "final_text_correct",
        "suggestions",
        "character_edit_distance",
        "correction_top_1",
        "correction_top_3",
        "error_type",
        "status",
        "error",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(metrics, output_path):
    successful = metrics["successful"]
    spelling_evaluated = metrics["spelling_evaluated"]
    incorrect_words = metrics["incorrect_words"]

    character_accuracy = safe_division(
        max(
            0,
            metrics["expected_characters"] - metrics["character_distance"],
        ),
        metrics["expected_characters"],
    )

    segmentation_accuracy = safe_division(
        metrics["correct_segmentations"],
        successful,
    )
    raw_word_accuracy = safe_division(
        metrics["correct_raw_words"],
        successful,
    )
    final_text_accuracy = safe_division(
        metrics["correct_final_texts"],
        successful,
    )
    spelling_accuracy = safe_division(
        metrics["correct_spelling_detections"],
        spelling_evaluated,
    )

    precision = safe_division(
        metrics["true_positive"],
        metrics["true_positive"] + metrics["false_positive"],
    )
    recall = safe_division(
        metrics["true_positive"],
        metrics["true_positive"] + metrics["false_negative"],
    )
    f1_score = safe_division(
        2 * precision * recall,
        precision + recall,
    )

    correction_top_1 = safe_division(
        metrics["correct_corrections_top_1"],
        incorrect_words,
    )
    correction_top_3 = safe_division(
        metrics["correct_corrections_top_3"],
        incorrect_words,
    )

    print("\nEvaluation completed")
    print(f"Samples: {metrics['total']}")
    print(f"Successfully processed: {successful}")
    print(f"Execution errors: {metrics['failed']}")
    print(f"Segmentation accuracy: {segmentation_accuracy:.2%}")
    print(f"Raw character accuracy: {character_accuracy:.2%}")
    print(f"Raw word accuracy: {raw_word_accuracy:.2%}")
    print(f"Final text accuracy: {final_text_accuracy:.2%}")

    if spelling_evaluated:
        print(f"Spelling samples: {spelling_evaluated}")
        print(f"Spelling detection accuracy: {spelling_accuracy:.2%}")
        print(f"Error detection precision: {precision:.2%}")
        print(f"Error detection recall: {recall:.2%}")
        print(f"Error detection F1: {f1_score:.2%}")
        print(f"Correction Top-1: {correction_top_1:.2%}")
        print(f"Correction Top-3: {correction_top_3:.2%}")

    print(f"Detailed results: {output_path}")


def sample_value(sample, key, default=""):
    return (sample.get(key) or default).strip()


def main():
    arguments = parse_arguments()

    if not 1 <= arguments.top_k <= len(LETTERS):
        raise ValueError(f"--top-k must be between 1 and {len(LETTERS)}")

    if arguments.beam_width < 1:
        raise ValueError("--beam-width must be positive")

    if arguments.max_edit_distance < 0:
        raise ValueError("--max-edit-distance cannot be negative")

    if not 0 <= arguments.preserve_raw_threshold <= 1:
        raise ValueError("--preserve-raw-threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = load_model(arguments.checkpoint, device)
    corrector = None

    if arguments.language == "it":
        custom_dictionary = (
            arguments.custom_dictionary
            if arguments.custom_dictionary and arguments.custom_dictionary.exists()
            else None
        )
        corrector = ItalianSpellCorrector(
            arguments.dictionary,
            max_edit_distance=arguments.max_edit_distance,
            custom_dictionary_path=custom_dictionary,
            extra_dictionary_paths=resolve_extra_dictionaries(
                arguments.extra_dictionary
            ),
        )

    samples = load_manifest(arguments.manifest)

    metrics = {
        "total": len(samples),
        "successful": 0,
        "failed": 0,
        "correct_segmentations": 0,
        "expected_characters": 0,
        "character_distance": 0,
        "correct_raw_words": 0,
        "correct_final_texts": 0,
        "spelling_evaluated": 0,
        "correct_spelling_detections": 0,
        "incorrect_words": 0,
        "correct_corrections_top_1": 0,
        "correct_corrections_top_3": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }

    results = []

    for sample_index, sample in enumerate(samples, start=1):
        image_path = resolve_image_path(
            sample_value(sample, "image"),
            arguments.images_dir,
        )
        expected_text = sample_value(sample, "expected_text")
        expected_correction = sample_value(sample, "expected_correction")
        expected_final = sample_value(sample, "expected_final")
        category = sample_value(sample, "category", "uncategorized")

        try:
            expected_is_correct = parse_boolean(sample_value(sample, "is_correct"))
            evaluate_spelling = parse_optional_boolean(
                sample.get("evaluate_spelling"),
                default=True,
            )

            if not expected_text:
                raise ValueError("expected_text cannot be empty")

            if expected_is_correct and not expected_correction:
                expected_correction = expected_text

            if (
                evaluate_spelling
                and not expected_is_correct
                and not expected_correction
            ):
                raise ValueError(
                    "Incorrect words require expected_correction when spelling "
                    "evaluation is enabled"
                )

            if not expected_final:
                if not expected_is_correct and expected_correction:
                    expected_final = expected_correction
                else:
                    expected_final = expected_text

            prediction = predict_word(
                image_path=image_path,
                model=model,
                device=device,
                corrector=corrector,
                top_k=arguments.top_k,
                beam_width=arguments.beam_width,
            )

            correction_suppressed = should_suppress_correction(
                correction=prediction["correction"],
                raw_word=prediction["raw_word"],
                average_confidence=prediction["average_confidence"],
                segmentation_uncertain=prediction["segmentation_uncertain"],
                correction_mode=arguments.correction_mode,
                preserve_raw_threshold=arguments.preserve_raw_threshold,
            )

            evaluation = evaluate_prediction(
                prediction=prediction,
                expected_text=expected_text,
                expected_is_correct=expected_is_correct,
                expected_correction=expected_correction,
                expected_final=expected_final,
                evaluate_spelling=evaluate_spelling,
                correction_suppressed=correction_suppressed,
            )

            correction = prediction["correction"]
            suggested_correction = ""

            if correction is not None and correction.correction:
                suggested_correction = correction.correction

            metrics["successful"] += 1
            metrics["expected_characters"] += len(normalize_word(expected_text))
            metrics["character_distance"] += evaluation["character_distance"]

            if evaluation["segmentation_ok"]:
                metrics["correct_segmentations"] += 1

            if evaluation["raw_word_correct"]:
                metrics["correct_raw_words"] += 1

            if evaluation["final_text_correct"]:
                metrics["correct_final_texts"] += 1

            if evaluate_spelling:
                metrics["spelling_evaluated"] += 1

                if evaluation["spelling_detection_correct"]:
                    metrics["correct_spelling_detections"] += 1

                expected_error = not expected_is_correct
                predicted_error = not evaluation["predicted_is_correct"]

                if expected_error and predicted_error:
                    metrics["true_positive"] += 1
                elif not expected_error and predicted_error:
                    metrics["false_positive"] += 1
                elif expected_error and not predicted_error:
                    metrics["false_negative"] += 1

                if not expected_is_correct:
                    metrics["incorrect_words"] += 1

                    if evaluation["correction_top_1"]:
                        metrics["correct_corrections_top_1"] += 1

                    if evaluation["correction_top_3"]:
                        metrics["correct_corrections_top_3"] += 1

            results.append(
                {
                    "image": str(image_path),
                    "category": category,
                    "expected_text": expected_text,
                    "expected_final": expected_final,
                    "evaluate_spelling": evaluate_spelling,
                    "expected_is_correct": expected_is_correct,
                    "expected_correction": expected_correction,
                    "segment_count": prediction["segment_count"],
                    "expected_length": len(normalize_word(expected_text)),
                    "segmentation_ok": evaluation["segmentation_ok"],
                    "segmentation_uncertain": prediction["segmentation_uncertain"],
                    "raw_ocr": prediction["raw_word"],
                    "raw_word_correct": evaluation["raw_word_correct"],
                    "ocr_candidates": format_candidates(prediction["ocr_candidates"]),
                    "average_confidence": f"{prediction['average_confidence']:.4f}",
                    "predicted_is_correct": evaluation["predicted_is_correct"],
                    "spelling_detection_correct": evaluation[
                        "spelling_detection_correct"
                    ],
                    "suggested_correction": suggested_correction,
                    "correction_suppressed": correction_suppressed,
                    "final_text": evaluation["final_text"],
                    "final_text_correct": evaluation["final_text_correct"],
                    "suggestions": format_suggestions(correction),
                    "character_edit_distance": evaluation["character_distance"],
                    "correction_top_1": evaluation["correction_top_1"],
                    "correction_top_3": evaluation["correction_top_3"],
                    "error_type": evaluation["error_type"],
                    "status": "OK",
                    "error": "",
                }
            )

            print(
                f"[{sample_index}/{len(samples)}] "
                f"{image_path.name}: "
                f"raw={prediction['raw_word']} | "
                f"final={evaluation['final_text']} "
                f"[{evaluation['error_type']}]"
            )

        except Exception as error:
            metrics["failed"] += 1

            results.append(
                {
                    "image": str(image_path),
                    "category": category,
                    "expected_text": expected_text,
                    "expected_final": expected_final,
                    "evaluate_spelling": sample.get("evaluate_spelling", ""),
                    "expected_is_correct": sample.get("is_correct", ""),
                    "expected_correction": expected_correction,
                    "segment_count": "",
                    "expected_length": "",
                    "segmentation_ok": "",
                    "segmentation_uncertain": "",
                    "raw_ocr": "",
                    "raw_word_correct": "",
                    "ocr_candidates": "",
                    "average_confidence": "",
                    "predicted_is_correct": "",
                    "spelling_detection_correct": "",
                    "suggested_correction": "",
                    "correction_suppressed": "",
                    "final_text": "",
                    "final_text_correct": "",
                    "suggestions": "",
                    "character_edit_distance": "",
                    "correction_top_1": "",
                    "correction_top_3": "",
                    "error_type": "EXECUTION_ERROR",
                    "status": "ERROR",
                    "error": str(error),
                }
            )

            print(
                f"[{sample_index}/{len(samples)}] "
                f"{image_path.name}: ERROR - {error}"
            )

    write_results(arguments.output, results)
    print_summary(metrics, arguments.output)

    if metrics["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
