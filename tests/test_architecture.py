"""Architectural boundary and isolation tests for DocuChat.

These tests use Python's Abstract Syntax Tree (ast) to verify that layer boundaries
are strictly maintained across the frontend, API contracts, configuration, and business logic.
"""

import ast
from pathlib import Path
from typing import List, Set
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_all_imported_modules(file_path: Path) -> Set[str]:
    """Parse a Python source file using AST and return a set of all imported module names."""
    if not file_path.exists():
        pytest.fail(f"Source file not found: {file_path}")

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    imported_modules: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    return imported_modules


# ---------------------------------------------------------------------------
# Architecture Boundary Tests
# ---------------------------------------------------------------------------


def test_schemas_has_no_business_logic_dependencies():
    """Verify backend/schemas.py contains only data contracts and no business logic or main imports.

    Why this rule matters:
    Schemas define API request/response contracts. They should be clean Pydantic models
    with zero coupling to database, RAG engine, or route logic.
    """
    schemas_file = ROOT_DIR / "backend" / "schemas.py"
    imports = get_all_imported_modules(schemas_file)

    disallowed = ["backend.rag_service", "backend.main", "rag_service", "main"]
    for mod in imports:
        for forbidden in disallowed:
            assert not mod.startswith(forbidden), (
                f"backend/schemas.py violates layering: imports forbidden '{mod}'"
            )


def test_config_has_no_service_or_route_dependencies():
    """Verify backend/config.py contains only settings and no service or route dependencies.

    Why this rule matters:
    Configuration is foundational and must not depend on higher-level business services
    or FastAPI endpoints, preventing circular import cycles.
    """
    config_file = ROOT_DIR / "backend" / "config.py"
    imports = get_all_imported_modules(config_file)

    disallowed = ["backend.rag_service", "backend.main", "rag_service", "main"]
    for mod in imports:
        for forbidden in disallowed:
            assert not mod.startswith(forbidden), (
                f"backend/config.py violates layering: imports forbidden '{mod}'"
            )


def test_frontend_has_no_direct_backend_or_ai_imports():
    """Verify frontend/app.py does not import backend packages, Chroma, or LangChain directly.

    Why this rule matters:
    The frontend must remain completely decoupled from backend internals and database models.
    It must communicate strictly across the HTTP network boundary via the `requests` library.
    """
    frontend_file = ROOT_DIR / "frontend" / "app.py"
    imports = get_all_imported_modules(frontend_file)

    disallowed_prefixes = [
        "backend",
        "chromadb",
        "langchain",
        "langchain_community",
        "langchain_google_genai",
        "langchain_chroma",
        "google.generativeai",
        "google.genai",
    ]

    for mod in imports:
        for forbidden in disallowed_prefixes:
            assert not mod == forbidden and not mod.startswith(f"{forbidden}."), (
                f"frontend/app.py violates client-server boundary: directly imports '{mod}'. "
                "Frontend must only communicate with backend via HTTP."
            )


def test_rag_service_has_no_frontend_dependencies():
    """Verify backend/rag_service.py has no dependency on the frontend presentation layer.

    Why this rule matters:
    Core business logic and vector retrieval algorithms should never depend on UI code.
    """
    rag_file = ROOT_DIR / "backend" / "rag_service.py"
    imports = get_all_imported_modules(rag_file)

    for mod in imports:
        assert not mod == "frontend" and not mod.startswith("frontend."), (
            f"backend/rag_service.py violates layer hierarchy: imports '{mod}' from frontend."
        )
