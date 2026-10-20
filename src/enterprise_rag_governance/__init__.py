"""Enterprise RAG Governance.

A clean-room reference implementation of permission-aware
retrieval-augmented generation: documents carry ACLs and sensitivity
levels, ingestion redacts PII, retrieval filters by the requesting
principal *before* ranking, answers are extractive with chunk-level
citations (and abstain without authorized evidence), and every
retrieval is audited. All data is synthetic; the embedder and the
answerer are deterministic and fully offline — no API keys, no LLM.
"""

__version__ = "0.1.0"
