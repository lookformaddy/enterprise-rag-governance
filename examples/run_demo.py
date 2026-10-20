"""Run the end-to-end demo offline: ingest, redact, retrieve, answer, eval.

Usage: python examples/run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprise_rag_governance.demo import run_demo  # noqa: E402

if __name__ == "__main__":
    result = run_demo()
    print()
    print(f"eval leak rate: {result['eval']['unauthorized_leak_rate']} (must be 0)")
    print(f"audit entries: {result['audit_entries']}")
    assert result["eval"]["unauthorized_leak_rate"] == 0.0
    print("Demo OK — authorized answer cited, unauthorized query abstained, no leaks.")
