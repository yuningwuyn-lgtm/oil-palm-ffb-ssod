from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript_jae" / "figures"


def flatten_png(path: Path) -> bool:
    image = Image.open(path)
    if image.mode not in {"RGBA", "LA"}:
        return False
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    white.convert("RGB").save(path, optimize=True)
    return True


def main():
    changed = []
    for path in sorted(FIG_DIR.glob("*.png")):
        if flatten_png(path):
            changed.append(path.name)
    print({"flattened": changed, "count": len(changed)})


if __name__ == "__main__":
    main()
