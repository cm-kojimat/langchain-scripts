# langchain-scripts

`langchain-scripts` is a Python package that provides a command-line interface (CLI) for interacting with language models and vector databases using the LangChain library.

## Installation

You can install `langchain-scripts` using Poetry:

```bash
poetry install
```

## Usage

The main entry point for the CLI is the `lchain` command. You can run it with the following options:

```
usage: lchain.py [-h] [--model MODEL] [--documents DOCUMENTS] [--system SYSTEM] [--verbose VERBOSE] [--debug DEBUG]

options:
  -h, --help            show this help message and exit
  --model MODEL         The URL scheme for the chat model (default: ollama://llama2)
  --documents DOCUMENTS
                        The URL scheme for the document loader
  --system SYSTEM       The system prompt for the chat model
  --verbose VERBOSE     Enable verbose logging
  --debug DEBUG         Enable debug logging
```

### Chat Model

The `--model` option specifies the chat model to use. The format is a URL scheme that includes the model type and name. For example:

- `ollama://llama2`: Use the OllaMa chat model with the `llama2` model.
- `openai://gpt-3.5-turbo`: Use the OpenAI chat model with the `gpt-3.5-turbo` model.
- `bedrock://my-model?region=us-west-2`: Use the Bedrock chat model with the `my-model` model in the `us-west-2` region.

### Document Loader

The `--documents` option specifies the document loader to use. The format is a URL scheme that includes the loader type and additional parameters. For example:

- `ollama://llama2/documents_dir?glob=*.md`: Load Markdown files from the specified directory.

### System Prompt

The `--system` option sets the system prompt for the chat model. This can be used to provide context or instructions for the model.

### Verbose and Debug Logging

The `--verbose` and `--debug` options enable verbose and debug logging, respectively.

### Interactive Mode

If you run `lchain` without piping input from stdin, it will enter an interactive mode where you can input prompts and receive responses from the chat model.

### Non-interactive Mode

If you pipe input to `lchain` from stdin, it will process the input and print the response without entering interactive mode.

## Development

You can use the following Make commands for development:

- `make format`: Format the code using Black and Ruff.
- `make lint`: Check the code for linting errors using Black, Ruff, and Pyright.

## License

This project is licensed under the [MIT License](LICENSE).
