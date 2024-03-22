import logging
import os
import sys
from argparse import ArgumentParser

from langchain.globals import set_debug, set_verbose

from langchain_scripts.core import build_chain


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("LCHAIN_CHAT_MODEL", "ollama://llama2"),
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=os.environ.get("LCHAIN_EMBEDDING", "ollama://llama2"),
    )
    parser.add_argument("--system", type=str, default="")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        set_verbose(args.verbose)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        set_debug(args.debug)

    chain = build_chain(model=args.model)

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        invoke_input = {
            "system": args.system,
            "input": text,
            "embedding": args.embedding,
        }
        resp = chain.invoke(input=invoke_input)
        print(resp["answer"])
        return

    while True:
        text = input("Input: ")
        if not text:
            break
        invoke_input = {
            "system": args.system,
            "input": text,
            "embedding": args.embedding,
        }
        resp = chain.invoke(input=invoke_input)
        print("Answer: ", resp["answer"])
        if resp.get("documents"):
            print(
                "Document: ",
                [doc.metadata["source"] for doc in resp["documents"]],
            )


if __name__ == "__main__":
    main()
