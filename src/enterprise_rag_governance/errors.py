"""Domain errors with stable codes."""

from __future__ import annotations


class RagError(Exception):
    """Base error for the platform."""

    code = "rag_error"


class ValidationError(RagError):
    code = "validation_error"


class DocumentNotFoundError(RagError):
    code = "document_not_found"


class DuplicateDocumentError(RagError):
    code = "duplicate_document"
