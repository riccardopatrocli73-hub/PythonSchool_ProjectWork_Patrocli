"""Word segmentation into individual characters"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segments"


# Character segment definition
@dataclass(frozen=True)
class CharacterSegment:
    box: tuple[int, int, int, int]  # coordinates
    uncertain: bool = False  # True if uncertain, False if not
    reason: str = ""  # Explain the meaning of segmentation


def _segment_center_y(segment):
    return (segment.box[1] + segment.box[3]) / 2


# Segments division in lines
def group_segments_into_lines(segments):
    """Group character boxes into text lines in reading order."""

    if not segments:
        return []

    # Typical height computation
    heights = [max(1, segment.box[3] - segment.box[1]) for segment in segments]
    typical_height = float(np.median(heights))
    # ch ordering top to bottom and left to right
    ordered_segments = sorted(
        segments,
        key=lambda segment: (_segment_center_y(segment), segment.box[0]),
    )
    lines = []

    # Check if a ch is in a line or in another based on the alignment with line center
    for segment in ordered_segments:
        center_y = _segment_center_y(segment)
        best_line = None
        best_distance = float("inf")

        for line in lines:
            line_center = float(np.median([_segment_center_y(item) for item in line]))
            line_top = min(item.box[1] for item in line)
            line_bottom = max(item.box[3] for item in line)
            segment_top = segment.box[1]
            segment_bottom = segment.box[3]
            overlap = max(
                0,
                min(line_bottom, segment_bottom) - max(line_top, segment_top),
            )
            minimum_height = max(
                1,
                min(line_bottom - line_top, segment_bottom - segment_top),
            )
            overlap_ratio = overlap / minimum_height
            center_distance = abs(center_y - line_center)

            if (
                overlap_ratio >= 0.25 or center_distance <= typical_height * 0.60
            ) and center_distance < best_distance:
                best_line = line
                best_distance = center_distance

        if best_line is None:
            lines.append([segment])
        else:
            best_line.append(segment)

    # Line ordering top to bottom, ch ordering left to right
    lines.sort(
        key=lambda line: float(
            np.median([_segment_center_y(segment) for segment in line])
        )
    )

    for line in lines:
        line.sort(key=lambda segment: segment.box[0])

    return lines


# Phrase division into words
def _group_line_segments_into_words(segments):
    if len(segments) <= 1:
        return [list(segments)] if segments else []

    # Typical width computation
    widths = [max(1, segment.box[2] - segment.box[0]) for segment in segments]
    typical_width = float(np.median(widths))
    # Gaps computation: based on its value, a ch is in a word or not
    gaps = [
        segments[index + 1].box[0] - segments[index].box[2]
        for index in range(len(segments) - 1)
    ]
    positive_gaps = sorted(gap for gap in gaps if gap >= 0)

    if positive_gaps:
        lower_gap_count = max(1, int(len(positive_gaps) * 0.60))
        typical_letter_gap = float(np.median(positive_gaps[:lower_gap_count]))
    else:
        typical_letter_gap = 0.0

    # Gap threshold definition
    word_gap_threshold = max(
        typical_width * 0.60,
        typical_letter_gap * 1.30,
    )
    groups = [[segments[0]]]

    # If gap > threshold, new word beginning
    for current, gap in zip(segments[1:], gaps):
        if gap > word_gap_threshold:
            groups.append([current])
        else:
            groups[-1].append(current)

    return groups


# Combination of previous steps
def group_segments_into_text(segments):
    """
    1) Division into lines
    2) Lines division into words
    Returns lines -> words -> characters
    """

    return [
        _group_line_segments_into_words(line)
        for line in group_segments_into_lines(segments)
    ]


def group_segments_into_words(segments):

    return [word for line in group_segments_into_text(segments) for word in line]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Handwritten word segmentation")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


# Dots and small marks management
def _merge_diacritics(components, minimum_area, minimum_height):
    main_components = []
    small_components = []
    minimum_small_area = max(3, round(minimum_area * 0.20))

    for component in components:
        if component["area"] >= minimum_area and component["height"] >= minimum_height:
            main_components.append(component)
        elif component["area"] >= minimum_small_area:
            small_components.append(component)

    for small in small_components:
        """
        For each small component, find the nearest main component to be merged.
        Checks:
        - Horizontal alignement
        - If above main component
        - If close enough to main component
        - If right dimensions
        """
        small_center_x = small["x"] + small["width"] / 2
        best_index = None
        best_distance = float("inf")

        for index, main in enumerate(main_components):
            horizontal_margin = max(3, round(main["width"] * 0.25))
            horizontal_match = (
                main["x"] - horizontal_margin
                <= small_center_x
                <= main["x"] + main["width"] + horizontal_margin
            )

            small_bottom = small["y"] + small["height"]
            vertical_gap = max(0, main["y"] - small_bottom)
            above_main_body = small["y"] < (main["y"] + main["height"] * 0.35)
            close_enough = vertical_gap <= max(
                5,
                round(main["height"] * 0.60),
            )
            plausible_size = small["width"] <= max(
                12,
                main["width"] * 1.25,
            )

            if not (
                horizontal_match and above_main_body and close_enough and plausible_size
            ):
                continue

            main_center_x = main["x"] + main["width"] / 2
            distance = abs(main_center_x - small_center_x) + vertical_gap

            if distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            continue

        # If a match is found, increase main component box dimensions
        # including the dot/mark
        main = main_components[best_index]
        left = min(main["x"], small["x"])
        top = min(main["y"], small["y"])
        right = max(
            main["x"] + main["width"],
            small["x"] + small["width"],
        )
        bottom = max(
            main["y"] + main["height"],
            small["y"] + small["height"],
        )

        main_components[best_index] = {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
            "area": main["area"] + small["area"],
        }

    return main_components


# Character end finding
def _find_split_column(binary, box, typical_width):
    # This function is needed when two letters are close and
    # OpenCV sees them togheter
    left, top, right, bottom = box
    component = binary[top:bottom, left:right] > 0
    height, width = component.shape

    minimum_part_width = max(
        3,
        round(min(height, typical_width) * 0.22),
    )

    if width < minimum_part_width * 2 + 1:
        return None

    # vertical projection computation, which says how much ink is in a column
    projection = component.sum(axis=0)
    maximum_ink = int(projection.max(initial=0))
    total_ink = int(projection.sum())

    if maximum_ink == 0 or total_ink == 0:
        return None

    maximum_valley_ink = max(2, round(maximum_ink * 0.25))
    candidates = []

    for column in range(
        minimum_part_width,
        width - minimum_part_width,
    ):
        valley_ink = int(projection[column])

        left_window = projection[max(0, column - 2) : column]
        right_window = projection[column + 1 : min(width, column + 3)]
        local_peak = max(
            int(left_window.max(initial=0)),
            int(right_window.max(initial=0)),
        )

        if valley_ink > maximum_valley_ink:
            continue

        if local_peak and valley_ink > local_peak * 0.55:
            continue

        left_ink = int(projection[:column].sum())
        right_ink = int(projection[column + 1 :].sum())
        ink_balance = min(left_ink, right_ink) / total_ink

        # A connector tail has little ink on one side and must be trimmed,
        # not interpreted as a second character.
        if ink_balance < 0.20:
            continue

        left_width = column
        right_width = width - column - 1
        width_penalty = (
            abs(left_width - typical_width) + abs(right_width - typical_width)
        ) / max(typical_width, 1)
        center_distance = abs(column - width / 2) / width
        candidates.append(
            (
                valley_ink,
                width_penalty,
                -ink_balance,
                center_distance,
                column,
            )
        )

    if not candidates:
        return None

    # Return best candidate
    candidates.sort()
    return left + candidates[0][4]


# Wide component splitting
def _split_wide_component(
    binary,
    box,
    typical_width,
    depth=0,
):
    """
    Recursive function:
    1) Width and height computation
    2) Threshold definition
    3) If wide component, column cutting
    4) If no valid cut, box marked as uncertain
    5) If valid cut, cut and function recall
    """
    left, top, right, bottom = box
    width = right - left
    height = bottom - top

    width_limit = max(
        height * 1.35,
        typical_width * 1.35,
    )

    if width <= width_limit:
        return [(box, False, "")]

    split_column = _find_split_column(
        binary,
        box,
        typical_width,
    )

    if split_column is None:
        return [(box, True, "wide_component")]

    left_box = (left, top, split_column, bottom)
    right_box = (split_column + 1, top, right, bottom)

    # Recursions limit. If depth >= 3, uncertain box
    if depth >= 3:
        return [
            (left_box, True, "split_depth_limit"),
            (right_box, True, "split_depth_limit"),
        ]

    return _split_wide_component(
        binary,
        left_box,
        typical_width,
        depth=depth + 1,
    ) + _split_wide_component(
        binary,
        right_box,
        typical_width,
        depth=depth + 1,
    )


# Small margin adding (8% of max dim) in box environment
def _pad_box(box, image_width, image_height):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    padding = max(2, round(max(width, height) * 0.08))

    return (
        max(0, left - padding),
        max(0, top - padding),
        min(image_width, right + padding),
        min(image_height, bottom + padding),
    )


# Line height estimation
def _estimate_text_line_height(binary):
    # Finds rows with ink in the binary image
    occupied_rows = np.flatnonzero(np.any(binary > 0, axis=1))

    if len(occupied_rows) == 0:
        return binary.shape[0]

    # If a big gap is detected, 2 lines
    maximum_internal_gap = max(3, round(binary.shape[0] * 0.03))
    spans = []
    start = int(occupied_rows[0])
    previous = start

    for row in occupied_rows[1:]:
        row = int(row)

        if row - previous > maximum_internal_gap:
            spans.append(previous - start + 1)
            start = row

        previous = row

    spans.append(previous - start + 1)

    maximum_span = max(spans)
    significant_spans = [span for span in spans if span >= maximum_span * 0.40]

    return max(1, round(float(np.median(significant_spans))))


# Main function
def segment_characters(image, choose_split=None):
    """Segment ink; an optional callback can reject proposed wide-letter cuts.

    The callback receives a padded whole CharacterSegment and padded parts,
    and returns True to accept the split. Without it, geometry decides alone.
    """
    # Image conversion to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Light blur addition to remove noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        sigmaX=0,
    )

    # Dark writing becomes white over a black background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove borders around images
    binary[:2, :] = 0
    binary[-2:, :] = 0
    binary[:, :2] = 0
    binary[:, -2:] = 0

    # Preserve narrow gaps between letters. A 3x3 closing can join distinct
    # characters in small images before connected-component detection.

    component_count, _, statistics, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    # Minimum threshold estimation to filter noise
    image_height, image_width = binary.shape
    text_line_height = _estimate_text_line_height(binary)

    minimum_area = max(
        15,
        round(text_line_height * image_width * 0.0002),
    )
    minimum_height = max(
        5,
        round(text_line_height * 0.20),
    )

    components = []

    # Component zero is the background
    for component in range(1, component_count):
        x, y, width, height, area = statistics[component]

        if area < max(3, round(minimum_area * 0.20)):
            continue

        # Ignoring components that cover almost the entire image
        if width > image_width * 0.95 and height > image_height * 0.95:
            continue

        # Each component is stored as a dictionary
        components.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
            }
        )

    components = _merge_diacritics(
        components,
        minimum_area,
        minimum_height,
    )

    if not components:
        raise ValueError("No character detected. Check the image and threshold.")

    # Components to boxes conversion
    raw_boxes = [
        (
            component["x"],
            component["y"],
            component["x"] + component["width"],
            component["y"] + component["height"],
        )
        for component in components
    ]

    # Mean width estimation
    reliable_widths = [
        right - left
        for left, top, right, bottom in raw_boxes
        if (right - left) <= (bottom - top) * 1.35
    ]

    if reliable_widths:
        typical_width = float(np.median(reliable_widths))
    else:
        typical_width = float(
            np.median([(bottom - top) * 0.70 for _, top, _, bottom in raw_boxes])
        )

    split_boxes = []

    # Wide components splitting
    for box in raw_boxes:
        parts = _split_wide_component(binary, box, typical_width)
        if len(parts) > 1 and choose_split is not None:
            whole = CharacterSegment(_pad_box(box, image_width, image_height))
            alternatives = [
                CharacterSegment(
                    _pad_box(part_box, image_width, image_height), uncertain, reason
                )
                for part_box, uncertain, reason in parts
            ]
            if not choose_split(whole, alternatives):
                parts = [(box, False, "")]
        split_boxes.extend(parts)

    # Boxes ordered left to right
    split_boxes.sort(key=lambda item: item[0][0])

    segments = [
        CharacterSegment(
            box=_pad_box(
                box,
                image_width,
                image_height,
            ),
            uncertain=uncertain,
            reason=reason,
        )
        for box, uncertain, reason in split_boxes
    ]

    return binary, segments  # Binary: Image for debug; segments: found ch


# Debug or use from terminal
def save_results(image, binary, boxes, output_dir):
    # Output dir creation
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_image = image.copy()

    # Ch cut storing
    for position, segment in enumerate(
        boxes,
        start=1,
    ):

        left, top, right, bottom = segment.box

        character = image[top:bottom, left:right]

        character_path = output_dir / f"character_{position:02d}.png"
        cv2.imwrite(str(character_path), character)

        cv2.rectangle(
            annotated_image,
            (left, top),
            (right, bottom),
            color=(0, 0, 255),
            thickness=2,
        )

        cv2.putText(
            annotated_image,
            str(position),
            (left, max(20, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        print(f"Character {position}: " f"x = {left} : {right}, y = {top} : {bottom}")

    cv2.imwrite(
        str(output_dir / "binary.png"),
        binary,
    )
    cv2.imwrite(
        str(output_dir / "segmented.png"),
        annotated_image,
    )


# Main allows to use this file directly from terminal
def main():

    # Arguments reading
    arguments = parse_arguments()

    # Check if image exists
    if not arguments.image.exists():
        raise FileNotFoundError(f"Image not found: {arguments.image}")

    # Image loading with OpenCV
    image = cv2.imread(str(arguments.image))

    if image is None:
        raise ValueError(f"Unable to read image: {arguments.image}")

    output_dir = arguments.output_dir / arguments.image.stem

    # Ch Segmentation
    binary, boxes = segment_characters(image)

    # Result storing
    save_results(
        image=image,
        binary=binary,
        boxes=boxes,
        output_dir=output_dir,
    )

    # Ch printing
    print(f"\nDetected characters: {len(boxes)}")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
