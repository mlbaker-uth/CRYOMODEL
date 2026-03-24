#!/usr/bin/env python3
"""
Fix ChimeraX Bundle classifier in a built wheel so toolshed install accepts it.
Run after: devel build /path/to/chimerax-bundles/crymodel
Then:      python fix_wheel_metadata.py dist/cryomodel-*.whl
Then:      toolshed install dist/cryomodel-*.whl  (in ChimeraX)
"""
import re
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

# 7-field Bundle line the toolshed expects (categories :: min :: max :: api_module :: supercedes :: custom_init)
BUNDLE_LINE_7 = "Classifier: ChimeraX :: Bundle :: Volume Data,Structural Analysis :: 1 :: 1 :: chimerax.cryomodel :: :: false"


def fix_metadata(metadata_content: str) -> str:
    """Replace malformed ChimeraX :: Bundle line(s) with the correct 7-field line."""
    lines = metadata_content.splitlines()
    out = []
    bundle_replaced = False
    for line in lines:
        if line.startswith("Classifier: ChimeraX :: Bundle ::"):
            if not bundle_replaced:
                out.append(BUNDLE_LINE_7)
                bundle_replaced = True
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def fix_wheel(wheel_path: Path) -> None:
    wheel_path = wheel_path.resolve()
    if not wheel_path.exists():
        raise FileNotFoundError(wheel_path)
    if not zipfile.is_zipfile(wheel_path):
        raise ValueError(f"Not a zip/wheel: {wheel_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(wheel_path, "r") as z:
            z.extractall(tmp)
        # Find METADATA (e.g. cryomodel-0.1.0.dist-info/METADATA)
        dist_info = next(tmp.glob("*.dist-info"), None)
        if not dist_info or not dist_info.is_dir():
            raise ValueError("No .dist-info in wheel")
        meta = dist_info / "METADATA"
        if not meta.exists():
            raise ValueError("No METADATA in wheel")
        content = meta.read_text(encoding="utf-8")
        new_content = fix_metadata(content)
        meta.write_text(new_content, encoding="utf-8")
        # Repack wheel (same name)
        new_wheel = tmp / wheel_path.name
        with zipfile.ZipFile(new_wheel, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.rglob("*"):
                if f.is_file():
                    arcname = f.relative_to(tmp)
                    z.write(f, arcname)
        shutil.copy2(new_wheel, wheel_path)
    print(f"Fixed: {wheel_path}")


def main():
    if len(sys.argv) < 2:
        dist = Path(__file__).parent / "dist"
        wheels = list(dist.glob("*.whl")) if dist.exists() else []
        if not wheels:
            print("Usage: python fix_wheel_metadata.py <path-to-wheel>", file=sys.stderr)
            print("Example: python fix_wheel_metadata.py dist/cryomodel-0.1.0-py3-none-any.whl", file=sys.stderr)
            sys.exit(1)
        wheel_path = wheels[0]
        print(f"Using: {wheel_path}")
    else:
        wheel_path = Path(sys.argv[1])
    fix_wheel(wheel_path)


if __name__ == "__main__":
    main()
