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
        "-m",
        type=str,
        default=os.environ.get("LCHAIN_CHAT_MODEL", "ollama://llama2"),
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=str,
        default=os.environ.get("LCHAIN_EMBEDDING", None),
    )
    parser.add_argument("--system", "-s", type=str, default="")
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
    )
    parser.add_argument("--debug", "-d", default=False, action="store_true")
    parser.add_argument("--buffer", "-b", default=False, action="store_true")
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

    prev_context = {}
    while True:
        if args.buffer:
            sys.stderr.write("* (<Ctrl-d> is done.) Input: ")
            sys.stderr.flush()
            text = sys.stdin.read().strip()
        else:
            text = input("* Input: ")
        if not text:
            break
        sys.stderr.write("* Processing... *\n")
        sys.stderr.flush()
        invoke_input = {
            **prev_context,
            "system": args.system,
            "input": text,
            "embedding": args.embedding,
        }
        resp = chain.invoke(input=invoke_input)
        print("* Answer: ", resp["answer"])
        if resp.get("documents"):
            print(
                "* Document:\n",
                "\n".join(
                    [
                        f"""source: {doc.metadata["source"]}
content: {doc.page_content[:150]}
"""
                        for doc in resp["documents"]
                    ],
                ),
            )
        prev_context = resp


if __name__ == "__main__":
    main()
