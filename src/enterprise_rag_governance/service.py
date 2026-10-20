"""FastAPI service: ingest, query, audit, eval, health.

The service is a thin layer over the same objects the CLI and tests
use. Principals are supplied per request (in production this identity
would come from an OIDC token terminated in front of the service).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .demo import RagSystem, build_system
from .errors import DuplicateDocumentError
from .evaluation import GoldenQuestion
from .models import Document, Principal, Sensitivity


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    text: str
    owner: str
    sensitivity: str = "internal"
    groups: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrincipalPayload(BaseModel):
    user_id: str
    groups: list[str] = Field(default_factory=list)
    clearance: str = "internal"

    def to_principal(self) -> Principal:
        return Principal(user_id=self.user_id, groups=tuple(self.groups),
                         clearance=Sensitivity.parse(self.clearance))


class QueryRequest(BaseModel):
    principal: PrincipalPayload
    query: str
    k: int = 5


class EvalRequest(BaseModel):
    golden: list[dict[str, Any]]
    k: int = 5


def create_app(system: RagSystem | None = None) -> FastAPI:
    rag = system or build_system()
    app = FastAPI(title="Enterprise RAG Governance", version="0.1.0")
    app.state.system = rag

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "enterprise-rag-governance"}

    @app.post("/documents", status_code=201)
    def ingest(body: IngestRequest) -> dict[str, Any]:
        try:
            doc = Document(
                doc_id=body.doc_id, title=body.title, text=body.text,
                owner=body.owner, sensitivity=Sensitivity.parse(body.sensitivity),
                groups=tuple(body.groups), allowed_users=tuple(body.allowed_users),
                metadata=dict(body.metadata),
            )
            result = rag.ingestion.ingest(doc)
        except DuplicateDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/documents")
    def list_documents(principal_user: str = "anonymous") -> dict[str, Any]:
        # Listing is metadata-only and requires no clearance check here
        # because titles are not returned for restricted corpora in a
        # real deployment; this reference build returns ids + levels.
        return {"documents": [
            {"doc_id": d.doc_id, "sensitivity": d.sensitivity.name.lower()}
            for d in rag.store.documents()
        ], "requested_by": principal_user}

    @app.post("/query")
    def query(body: QueryRequest) -> dict[str, Any]:
        try:
            principal = body.principal.to_principal()
            retrieval = rag.retriever.retrieve(principal, body.query, k=body.k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        answer = rag.answerer.answer(body.query, retrieval)
        return {"answer": answer.to_dict(), "retrieval": retrieval.to_dict()}

    @app.get("/audit")
    def audit() -> dict[str, Any]:
        return {
            "retrievals": [e.to_dict() for e in rag.audit.entries()],
            "redactions": [e.to_dict() for e in rag.audit.redactions()],
        }

    @app.post("/eval")
    def run_eval(body: EvalRequest) -> dict[str, Any]:
        try:
            questions = [GoldenQuestion.from_dict(g) for g in body.golden]
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return rag.evaluator.evaluate(questions, k=body.k).to_dict()

    return app
