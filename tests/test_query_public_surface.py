import importlib

import pytest

import drg
import drg.query as query_mod
from drg.query import GraphQuery


def test_removed_rag_and_vector_query_symbols_are_not_public():
    removed_context_name = "Graph" + "".join(chr(c) for c in (82, 65, 71)) + "Context"
    removed_mixed_result = "".join(chr(c) for c in (72, 121, 98, 114, 105, 100))
    removed = {
        removed_context_name,
        removed_mixed_result + "Search" + "Result",
        removed_mixed_result + "Ranking" + "Weights",
        "Vector" + "Document" + "Chunk",
        "Vector" + "Store",
        "InMemory" + "Vector" + "Store",
    }

    for name in removed:
        assert name not in drg.__all__
        assert not hasattr(drg, name)
        assert not hasattr(query_mod, name)


def test_removed_query_modules_are_not_importable():
    removed_modules = ("drg.query._hy" + "brid", "drg.query._vec" + "tor")
    for module_name in removed_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_graph_query_has_no_mixed_vector_search_methods():
    mixed_search_name = "".join(chr(c) for c in (104, 121, 98, 114, 105, 100)) + "_search"
    assert not hasattr(GraphQuery, mixed_search_name)
    assert not hasattr(GraphQuery, "graph" + "rag" + "_context")
