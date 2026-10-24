"""Create a clean, small source handoff without environments, runtime data or secrets."""
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / ".sidecar" / "sidecar-gpu-source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [root / name for name in ("README.md", "pyproject.toml", "requirements.txt", "requirements.lock", "config.example.json", ".gitignore")]
    for name in ("sidecar_gpu", "tests", "examples", "scripts", "docs", "benchmarks"):
        for path in (root / name).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".token"):
                files.append(path)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path("sidecar-gpu") / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any(part in (".venv", ".sidecar", "pairing.token") for name in archive.namelist() for part in Path(name).parts)
    print(f"{output} ({output.stat().st_size:,} bytes, {len(files)} source files)")


if __name__ == "__main__":
    main()
