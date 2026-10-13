# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)

datas = [
    (str(project_root / "models" / "best_ch_cnn.pt"), "models"),
    (str(project_root / "models" / "best_ch_cnn_local_words.pt"), "models"),
    (
        str(project_root / "data" / "dictionaries" / "it_frequency.txt"),
        "data/dictionaries",
    ),
    (
        str(project_root / "data" / "dictionaries" / "en_frequency.txt"),
        "data/dictionaries",
    ),
    (
        str(project_root / "data" / "dictionaries" / "it_bigrams.txt"),
        "data/dictionaries",
    ),
    (
        str(project_root / "data" / "dictionaries" / "en_bigrams.txt"),
        "data/dictionaries",
    ),
]
binaries = []
hiddenimports = []

symspell_datas, symspell_binaries, symspell_hiddenimports = collect_all("symspellpy")
datas += symspell_datas
binaries += symspell_binaries
hiddenimports += symspell_hiddenimports

a = Analysis(
    [str(project_root / "src" / "qt_app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HandwrittenTextOCR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
