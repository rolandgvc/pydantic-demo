#!/usr/bin/env python3
"""CLI for deep research."""
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Load environment before other imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env', override=True)

# Suppress logfire warnings
import warnings
warnings.filterwarnings('ignore', message='Logfire')

# Configure logfire (+ Introspection export)
try:
    import logfire
    from introspection_sdk import IntrospectionSpanProcessor

    processor = IntrospectionSpanProcessor(service_name='deep-research')
    logfire.configure(
        service_name='deep-research',
        additional_span_processors=[processor],
    )
    logfire.instrument_pydantic_ai()
except Exception:
    # Keep CLI usable even when observability deps/config are missing
    pass

from .researcher import DeepResearcher


def print_status(msg: str):
    """Print status message."""
    print(f"\033[90m→ {msg}\033[0m", file=sys.stderr)


async def run_research(query: str, model: str, parallel: int, output_file: str = None):
    """Run deep research."""
    researcher = DeepResearcher(
        model=model,
        max_parallel_researchers=parallel,
        allow_clarification=False,
    )

    print(f"\n{'='*60}")
    print(f"  Deep Research")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")

    report = await researcher.research(query, on_status=print_status)

    print(f"\n{'='*60}")
    print(f"  REPORT")
    print(f"{'='*60}\n")
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n\033[90mReport saved to: {output_file}\033[0m")

    return report


async def interactive_mode(model: str, parallel: int):
    """Run in interactive mode."""
    print(f"\n{'='*60}")
    print(f"  Deep Research - Interactive Mode")
    print(f"{'='*60}")
    print(f"\nModel: {model}")
    print(f"Max parallel researchers: {parallel}")
    print("\nEnter your research query (or 'quit' to exit):\n")

    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ('quit', 'q', 'exit'):
                print("\nGoodbye!")
                break

            if not query:
                print("Please enter a query.\n")
                continue

            await run_research(query, model, parallel)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Deep Research CLI - Comprehensive AI-powered research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the latest developments in quantum computing?"
  %(prog)s "Compare React vs Vue vs Svelte for web development"
  %(prog)s -o report.md "History of artificial intelligence"
  %(prog)s -i  # Interactive mode
        """
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Research query'
    )
    parser.add_argument(
        '-m', '--model',
        default='openai:gpt-4o',
        help='Model to use (default: openai:gpt-4o)'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=3,
        help='Max parallel researchers (default: 3)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for report (markdown)'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    args = parser.parse_args()

    if args.interactive or not args.query:
        asyncio.run(interactive_mode(args.model, args.parallel))
    else:
        asyncio.run(run_research(args.query, args.model, args.parallel, args.output))


if __name__ == '__main__':
    main()
