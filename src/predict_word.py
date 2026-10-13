""" Recognition of a handwritten word. """

import argparse
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

from src.alphabet import LETTERS
from src.language_detection import (
    DEFAULT_DICTIONARY,
    DEFAULT_ENGLISH_DICTIONARY,
    analyze_languages,
    build_language_correctors,
    build_spell_corrector,
    format_language_name,
    language_score,
    select_language_analysis,
)
from src.predict_character import load_model
from src.preprocessing import preprocess_character
from src.segment_word import segment_characters
from src.spell_correct import build_ocr_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_ch_cnn.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "word_predictions"

# Parameters definition
def parse_arguments():
    parser = argparse.ArgumentParser(description="Recognize a handwritten word")

    parser.add_argument("image", type=Path)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--language",
        choices=("auto", "it", "en", "none"),
        default="auto",
        help="Language dictionary to use for spelling checks.",
    )

    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DEFAULT_DICTIONARY,
        help="Italian frequency dictionary.",
    )

    parser.add_argument(
        "--english-dictionary",
        type=Path,
        default=DEFAULT_ENGLISH_DICTIONARY,
        help="English frequency dictionary.",
    )

    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--correction-mode",
        choices=("aggressive", "conservative"),
        default="aggressive",
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

# Main of script to be run from terminal
def main():
    arguments = parse_arguments()

    if not arguments.image.exists():
        raise FileNotFoundError(f"Image not found: {arguments.image}")

    if not 1 <= arguments.top_k <= 26:
        raise ValueError("--top-k must be between 1 and 26")

    if arguments.beam_width < 1:
        raise ValueError("--beam-width must be positive")

    if arguments.candidate_count < 1:
        raise ValueError("--candidate-count must be positive")

    if not 0 <= arguments.confidence_threshold <= 1:
        raise ValueError("--confidence-threshold must be between 0 and 1")

    if not 0 <= arguments.preserve_raw_threshold <= 1:
        raise ValueError("--preserve-raw-threshold must be between 0 and 1")

    image = cv2.imread(str(arguments.image))

    if image is None:
        raise ValueError(f"Unable to read image: {arguments.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = load_model(
        arguments.checkpoint,
        device,
    )

    binary, boxes = segment_characters(image)

    output_dir = arguments.output_dir / arguments.image.stem
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_dir / "binary.png"),
        binary,
    )

    character_tensors = []

    for position, segment in enumerate(
        boxes,
        start=1,
    ):
        left, top, right, bottom = segment.box

        character = image[
            top:bottom,
            left:right,
        ]

        raw_character_path = output_dir / f"raw_character_{position:02d}.png"

        cv2.imwrite(
            str(raw_character_path),
            character,
        )

        tensor, processed_image = preprocess_character(
            character,
            trim_connectors=True,
        )

        character_tensors.append(tensor)

        processed_image.save(output_dir / f"processed_character_{position:02d}.png")

    character_batch = torch.cat(
        character_tensors,
        dim=0,
    ).to(device)

    with torch.inference_mode():
        logits = model(character_batch)

        probabilities = F.softmax(
            logits,
            dim=1,
        )

    top_probabilities, top_labels = torch.topk(
        probabilities,
        k=arguments.top_k,
        dim=1,
    )

    top_probabilities = top_probabilities.cpu()
    top_labels = top_labels.cpu()

    raw_word = "".join(LETTERS[int(label)] for label in top_labels[:, 0])

    ocr_candidates = build_ocr_candidates(
        top_probabilities,
        top_labels,
        beam_width=arguments.beam_width,
    )

    annotated_image = image.copy()
    uncertain_positions = []

    print("\nCharacter predictions:")

    for position, (
        segment,
        probability_values,
        label_values,
    ) in enumerate(
        zip(
            boxes,
            top_probabilities,
            top_labels,
        ),
        start=1,
    ):
        left, top, right, bottom = segment.box

        best_label = label_values[0].item()
        best_probability = probability_values[0].item()
        best_letter = LETTERS[best_label]

        if segment.uncertain or best_probability < arguments.confidence_threshold:
            uncertain_positions.append(position)
            status = "uncertain"
            box_color = (0, 165, 255)
        else:
            status = "accepted"
            box_color = (0, 180, 0)

        alternatives = []

        for probability, label in zip(
            probability_values,
            label_values,
        ):
            alternatives.append(
                f"{LETTERS[label.item()]}: " f"{probability.item():.2%}"
            )

        print(f"{position}. {best_letter} " f"({best_probability:.2%}, {status})")

        print(f"   Top-{arguments.top_k}: " f"{', '.join(alternatives)}")

        cv2.rectangle(
            annotated_image,
            (left, top),
            (right, bottom),
            box_color,
            thickness=2,
        )

        text = f"{best_letter} " f"{best_probability:.0%}"

        text_y = top - 8

        if text_y < 15:
            text_y = min(
                annotated_image.shape[0] - 5,
                bottom + 20,
            )

        cv2.putText(
            annotated_image,
            text,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            thickness=2,
        )

    average_confidence = top_probabilities[:, 0].mean().item()

    prediction_path = output_dir / "word_prediction.png"

    cv2.imwrite(
        str(prediction_path),
        annotated_image,
    )

    displayed_candidates = ocr_candidates[: arguments.candidate_count]

    print(f"\nDetected characters: {len(boxes)}")

    print(f"Raw OCR: {raw_word}")

    print(
        "OCR candidates: "
        + ", ".join(candidate.word for candidate in displayed_candidates)
    )

    print(f"Average confidence: " f"{average_confidence:.2%}")

    if arguments.language != "none":
        language_correctors = build_language_correctors(arguments)
        language_analyses = analyze_languages(
            raw_word,
            ocr_candidates,
            language_correctors,
        )
        selected_analysis, exact_matches = select_language_analysis(language_analyses)

        print(
            "Detected language: "
            + format_language_name(selected_analysis, exact_matches)
        )

        if selected_analysis is None:
            print("Spelling status: not checked")
            print("Suggested correction: none")
        else:
            correction = selected_analysis["result"]

            if correction.is_correct:
                print("Spelling status: correct")
                print("Suggested correction: none")
            elif correction.correction:
                print("Spelling status: possible error")
                print(f"Suggested correction: {correction.correction}")
            else:
                print("Spelling status: unknown word")
                print("Suggested correction: none")

            if correction.suggestions:
                print(
                    "Suggestions: "
                    + ", ".join(
                        suggestion.word for suggestion in correction.suggestions
                    )
                )

    if uncertain_positions:
        positions = ", ".join(str(position) for position in uncertain_positions)

        print("Uncertain character positions: " f"{positions}")
    else:
        print("All characters passed " "the confidence threshold.")

    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
