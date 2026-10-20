"""Command-line entry points.

Usage:
    enterprise-rag-governance demo
    enterprise-rag-governance eval
    enterprise-rag-governance serve --port 8000
"""

from __future__ import annotations

import argparse
import json


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    run_demo()
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .demo import GOLDEN, build_demo_system

    system = build_demo_system()
    print(json.dumps(system.evaluator.evaluate(GOLDEN, k=args.k).to_dict(), indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .service import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="enterprise-rag-governance", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the end-to-end offline demo")
    demo.set_defaults(func=_cmd_demo)

    ev = sub.add_parser("eval", help="Run the golden-question eval on the demo corpus")
    ev.add_argument("--k", type=int, default=5)
    ev.set_defaults(func=_cmd_eval)

    serve = sub.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
