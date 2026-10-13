"""Qt interface for handwritten text OCR."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from src.context_scoring import (
    choose_contextual_suggestion,
    select_dominant_language,
)
from src.text_recognizer import (
    DEFAULT_QT_CHECKPOINT,
    PROJECT_ROOT,
    TextRecognizer,
)

# Preserve the previous import surface for external callers.
WordRecognizer = TextRecognizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Qt app for handwritten text OCR")
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_QT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_qt():
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:
        raise RuntimeError(
            "PySide6 is required for the Qt interface. Install dependencies with "
            "`.project_venv/bin/python -m pip install -r requirements.txt`."
        ) from error

    return SimpleNamespace(
        Qt=Qt,
        QPixmap=QPixmap,
        QApplication=QApplication,
        QFileDialog=QFileDialog,
        QFrame=QFrame,
        QGridLayout=QGridLayout,
        QGroupBox=QGroupBox,
        QHBoxLayout=QHBoxLayout,
        QHeaderView=QHeaderView,
        QLabel=QLabel,
        QLineEdit=QLineEdit,
        QMainWindow=QMainWindow,
        QMessageBox=QMessageBox,
        QPushButton=QPushButton,
        QTableWidget=QTableWidget,
        QTableWidgetItem=QTableWidgetItem,
        QVBoxLayout=QVBoxLayout,
        QWidget=QWidget,
    )


def build_main_window(qt, recognizer, initial_image=None):
    class ImageViewer(qt.QLabel):
        def __init__(self):
            super().__init__("No image loaded")
            self.source_pixmap = None
            self.setAlignment(qt.Qt.AlignCenter)
            self.setMinimumSize(720, 360)
            self.setFrameShape(qt.QFrame.NoFrame)
            self.setStyleSheet(
                "QLabel { background: #f8fafc; border: 1px solid #d0d7de; "
                "border-radius: 8px; color: #667085; }"
            )

        def set_image(self, image_path):
            self.source_pixmap = qt.QPixmap(str(image_path))
            self._update_pixmap()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._update_pixmap()

        def _update_pixmap(self):
            if self.source_pixmap is None or self.source_pixmap.isNull():
                return

            target_size = self.size()
            scaled = self.source_pixmap.scaled(
                target_size,
                qt.Qt.KeepAspectRatio,
                qt.Qt.SmoothTransformation,
            )
            self.setPixmap(scaled)

    class MetricLabel(qt.QLabel):
        def __init__(self, text="-"):
            super().__init__(text)
            self.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
            self.setStyleSheet("font-weight: 600; color: #111827;")

    class MainWindow(qt.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Handwritten Text OCR")
            self.resize(1360, 820)
            self.setStyleSheet("""
                QMainWindow, QWidget { background: #f3f4f6; color: #111827; }
                QLineEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; }
                QPushButton { background: #1f6feb; color: white; border: 0; border-radius: 6px; padding: 8px 14px; font-weight: 600; }
                QPushButton:hover { background: #1a5fd0; }
                QGroupBox { background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; margin-top: 12px; padding: 12px; font-weight: 700; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
                QTableWidget { background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; gridline-color: #e5e7eb; alternate-background-color: #f8fafc; }
                QHeaderView::section { background: #eef2f7; color: #111827; border: 0; border-right: 1px solid #d0d7de; padding: 7px; font-weight: 700; }
                """)
            self.image_viewer = ImageViewer()
            self.path_input = qt.QLineEdit()
            self.browse_button = qt.QPushButton("Browse")
            self.analyze_button = qt.QPushButton("Analyze")
            self.correction_banner = qt.QLabel("Load an image to analyze handwriting")
            self.correction_banner.setAlignment(qt.Qt.AlignCenter)
            self.correction_banner.setMinimumHeight(82)
            self.correction_banner.setWordWrap(True)
            self.raw_text_label = MetricLabel()
            self.language_label = MetricLabel()
            self.status_label = MetricLabel()
            self.confidence_label = MetricLabel()
            self.suggestions_label = MetricLabel()
            self.detected_text_label = qt.QLabel("-")
            self.detected_text_label.setAlignment(qt.Qt.AlignCenter)
            self.detected_text_label.setWordWrap(True)
            self.detected_text_label.setTextInteractionFlags(
                qt.Qt.TextSelectableByMouse
            )
            self.detected_word_label = self.detected_text_label
            self.character_table = qt.QTableWidget(0, 6)
            self.character_table.setHorizontalHeaderLabels(
                ["#", "OCR", "Top 1", "Top 2", "Top 3", "Status"]
            )
            self.character_table.setAlternatingRowColors(True)
            self.character_table.verticalHeader().setVisible(False)
            self.character_table.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
            header = self.character_table.horizontalHeader()
            header.setSectionResizeMode(qt.QHeaderView.Interactive)
            self.character_table.setColumnWidth(0, 54)
            self.character_table.setColumnWidth(1, 54)
            self.character_table.setColumnWidth(2, 96)
            self.character_table.setColumnWidth(3, 96)
            self.character_table.setColumnWidth(4, 96)
            self.character_table.setColumnWidth(5, 84)
            header.setStretchLastSection(False)
            self.character_table.setMinimumHeight(320)
            self.character_table.setMaximumHeight(430)

            if initial_image:
                self.path_input.setText(str(initial_image))

            top_bar = qt.QHBoxLayout()
            top_bar.addWidget(self.path_input, 1)
            top_bar.addWidget(self.browse_button)
            top_bar.addWidget(self.analyze_button)

            image_box = qt.QGroupBox("Segmented image")
            image_layout = qt.QVBoxLayout()
            image_layout.addWidget(self.image_viewer, 1)
            image_box.setLayout(image_layout)

            summary_box = qt.QGroupBox("OCR summary")
            result_grid = qt.QGridLayout()
            result_grid.setHorizontalSpacing(16)
            result_grid.setVerticalSpacing(10)
            result_grid.addWidget(qt.QLabel("Raw text"), 0, 0)
            result_grid.addWidget(self.raw_text_label, 0, 1)
            result_grid.addWidget(qt.QLabel("Detected language"), 1, 0)
            result_grid.addWidget(self.language_label, 1, 1)
            result_grid.addWidget(qt.QLabel("Spelling status"), 2, 0)
            result_grid.addWidget(self.status_label, 2, 1)
            result_grid.addWidget(qt.QLabel("Average confidence"), 3, 0)
            result_grid.addWidget(self.confidence_label, 3, 1)
            result_grid.addWidget(qt.QLabel("Suggestions"), 4, 0)
            result_grid.addWidget(self.suggestions_label, 4, 1)
            summary_box.setLayout(result_grid)

            table_box = qt.QGroupBox("Character top-3")
            table_layout = qt.QVBoxLayout()
            table_layout.addWidget(self.character_table)
            table_box.setLayout(table_layout)

            detected_text_box = qt.QGroupBox("Detected Text")
            detected_text_layout = qt.QVBoxLayout()
            detected_text_layout.addWidget(self.detected_text_label)
            detected_text_box.setLayout(detected_text_layout)
            self.detected_text_label.setStyleSheet(
                "background: #f8fafc; color: #111827; border: 1px solid #d0d7de; "
                "border-radius: 8px; font-size: 24px; font-weight: 900; padding: 14px;"
            )

            side_layout = qt.QVBoxLayout()
            side_layout.addWidget(self.correction_banner)
            side_layout.addWidget(summary_box)
            side_layout.addWidget(table_box, 1)
            side_layout.addWidget(detected_text_box)

            content_layout = qt.QHBoxLayout()
            content_layout.setSpacing(16)
            content_layout.addWidget(image_box, 3)
            side_widget = qt.QWidget()
            side_widget.setLayout(side_layout)
            content_layout.addWidget(side_widget, 2)

            main_layout = qt.QVBoxLayout()
            main_layout.setContentsMargins(14, 14, 14, 14)
            main_layout.setSpacing(12)
            main_layout.addLayout(top_bar)
            main_layout.addLayout(content_layout, 1)

            container = qt.QWidget()
            container.setLayout(main_layout)
            self.setCentralWidget(container)

            self._set_banner_idle()
            self.browse_button.clicked.connect(self.browse_image)
            self.analyze_button.clicked.connect(self.analyze_image)

        def _set_banner_idle(self):
            self.correction_banner.setText("Load an image to analyze handwriting")
            self.correction_banner.setStyleSheet(
                "background: #eef2ff; color: #1e3a8a; border: 1px solid #c7d2fe; "
                "border-radius: 10px; font-size: 20px; font-weight: 800; padding: 14px;"
            )

        def _set_banner_for_result(self, result):
            if result.suggested_correction:
                self.correction_banner.setText(
                    f"Suggested text: {result.suggested_correction}"
                )
                self.correction_banner.setStyleSheet(
                    "background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; "
                    "border-radius: 10px; font-size: 24px; font-weight: 900; padding: 14px;"
                )
            else:
                self.correction_banner.setText("No correction needed")
                self.correction_banner.setStyleSheet(
                    "background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; "
                    "border-radius: 10px; font-size: 24px; font-weight: 900; padding: 14px;"
                )

        def closeEvent(self, event):
            reply = qt.QMessageBox.question(
                self,
                "Confirm exit",
                "Do you really want to close the application?",
                qt.QMessageBox.Yes | qt.QMessageBox.No,
                qt.QMessageBox.No,
            )

            if reply == qt.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

        def browse_image(self):
            path, _ = qt.QFileDialog.getOpenFileName(
                self,
                "Open handwritten text image",
                str(PROJECT_ROOT / "images"),
                "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
            )

            if path:
                self.path_input.setText(path)

        def analyze_image(self):
            try:
                result = recognizer.recognize_text(
                    Path(self.path_input.text()).expanduser()
                )
            except Exception as error:
                qt.QMessageBox.critical(self, "OCR error", str(error))
                return

            self._set_banner_for_result(result)
            self.raw_text_label.setText(result.raw_text)
            self.language_label.setText(result.detected_language)
            self.status_label.setText(result.spelling_status)
            self.confidence_label.setText(f"{result.average_confidence:.2%}")
            self.suggestions_label.setText(
                "; ".join(result.suggestions) if result.suggestions else "none"
            )
            self.detected_text_label.setText(result.final_text)
            self.image_viewer.set_image(result.annotated_image_path)
            self.character_table.setRowCount(len(result.character_predictions))

            multiple_words = len(result.word_results) > 1

            for row, prediction in enumerate(result.character_predictions):
                alternatives = prediction.alternatives + [("-", 0.0)] * 3
                position_label = (
                    f"{prediction.word_index}.{prediction.word_position}"
                    if multiple_words
                    else str(prediction.position)
                )
                values = [
                    position_label,
                    prediction.best_letter,
                    f"{alternatives[0][0]} {alternatives[0][1]:.2%}",
                    f"{alternatives[1][0]} {alternatives[1][1]:.2%}",
                    f"{alternatives[2][0]} {alternatives[2][1]:.2%}",
                    "uncertain" if prediction.uncertain else "accepted",
                ]

                for column, value in enumerate(values):
                    item = qt.QTableWidgetItem(value)
                    item.setToolTip(value)
                    item.setTextAlignment(qt.Qt.AlignCenter)
                    self.character_table.setItem(row, column, item)

                self.character_table.setRowHeight(row, 34)

    return MainWindow()


def main():
    arguments = parse_arguments()
    qt = load_qt()
    application = qt.QApplication(sys.argv)
    recognizer = TextRecognizer(
        checkpoint_path=arguments.checkpoint,
        top_k=arguments.top_k,
        beam_width=arguments.beam_width,
        confidence_threshold=arguments.confidence_threshold,
    )
    window = build_main_window(qt, recognizer, initial_image=arguments.image)
    window.show()

    if arguments.image:
        window.analyze_image()

    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
