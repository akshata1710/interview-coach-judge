"""Score one session file with Claude and print the evaluation as JSON.

    export ANTHROPIC_API_KEY=...
    python run.py examples/sample_session.json [--model claude-sonnet-5-5]
"""
import argparse
import asyncio
import json
from pathlib import Path

from judge import AnthropicClient, Session, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--model", default="claude-sonnet-5-5")
    args = parser.parse_args()

    session = Session.model_validate(json.loads(args.session.read_text()))
    result = asyncio.run(evaluate(session, AnthropicClient(model=args.model)))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
