# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['final_draft_maybe.py'],
    pathex=[],
    binaries=[],
    datas=[('attendance.xlsx', '.'), ('features/features.pkl', 'features'), ('C:\\Users\\aryan\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml', 'cv2\\data')],
    hiddenimports=[],
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
    name='final_draft_maybe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
