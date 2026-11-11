import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "evalgate" / "core"
SOURCE_FOLDERS = ("src", "tests", "scripts")

FORBIDDEN_PREFIXES = (
    "sqlalchemy",
    "asyncpg",
    "alembic",
    "fastapi",
    "starlette",
    "anthropic",
    "httpx",
    "requests",
    "aiohttp",
    "typer",
    "jinja2",
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for folder in SOURCE_FOLDERS:
        path = ROOT / folder
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def _imported_modules(path: Path) -> set[str]:
    # utf-8-sig so a stray BOM is reported by test_no_byte_order_marks rather
    # than crashing every other check in this file with a SyntaxError.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def _core_modules() -> list[Path]:
    return sorted(CORE.rglob("*.py"))


def test_core_package_exists() -> None:
    assert CORE.is_dir(), f"missing core package at {CORE}"


def test_no_byte_order_marks() -> None:
    """A utf-8 BOM breaks ast.parse and some tooling. Catch it as its own failure."""
    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_files()
        if path.read_bytes().startswith(b"\xef\xbb\xbf")
    ]
    assert not offenders, "utf-8 BOM found in: " + ", ".join(offenders)


def test_core_has_no_impure_imports() -> None:
    violations: list[str] = []
    for path in _core_modules():
        for module in _imported_modules(path):
            root = module.split(".")[0]
            if root in FORBIDDEN_PREFIXES:
                violations.append(f"{path.name} imports {module}")
    assert not violations, "core must stay pure: " + "; ".join(violations)


def test_core_does_not_import_other_evalgate_packages() -> None:
    violations: list[str] = []
    for path in _core_modules():
        for module in _imported_modules(path):
            if module.startswith("evalgate.") and not module.startswith("evalgate.core"):
                violations.append(f"{path.name} imports {module}")
    assert not violations, "core must not depend on outer layers: " + "; ".join(violations)