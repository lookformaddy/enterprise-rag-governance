"""Chunking: offsets are exact, windows overlap, ids are stable."""

import pytest

from enterprise_rag_governance.chunking import chunk_text

TEXT = ("Alpha sentence one. Beta sentence two is here. "
        "Gamma sentence three follows. Delta sentence four ends it.")


def test_single_chunk_for_short_text():
    chunks = chunk_text("doc", TEXT, chunk_chars=1000, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == TEXT
    assert (chunks[0].start_char, chunks[0].end_char) == (0, len(TEXT))


def test_offsets_match_source_text():
    chunks = chunk_text("doc", TEXT, chunk_chars=40, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert TEXT[c.start_char:c.end_char] == c.text


def test_chunk_ids_and_indexes():
    chunks = chunk_text("doc", TEXT, chunk_chars=40, overlap=10)
    assert [c.chunk_id for c in chunks] == [f"doc#c{i}" for i in range(len(chunks))]
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_shares_text():
    chunks = chunk_text("doc", TEXT, chunk_chars=40, overlap=10)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char < a.end_char  # windows overlap


def test_empty_text_yields_no_chunks():
    assert chunk_text("doc", "", chunk_chars=40, overlap=10) == []
    assert chunk_text("doc", "   ", chunk_chars=40, overlap=10) == []


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("doc", TEXT, chunk_chars=0)
    with pytest.raises(ValueError):
        chunk_text("doc", TEXT, chunk_chars=40, overlap=40)
