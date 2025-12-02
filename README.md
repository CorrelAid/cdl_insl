# InSl tools

## Prerequisites

- **Python 3.11+**
- **uv** - Python package manager ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

## Quick Start - Running Locally

### Step 1: Install uv

If you don't have uv installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd insl

# Install all dependencies (this may take a few minutes)
uv sync
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root with your OpenRouter API key:

```bash
# OpenRouter API Key (get from https://openrouter.ai)
OR_KEY="sk-or-v1-your-openrouter-key-here"
```

### Step 4: Run the Marimo Notebooks

Marimo is an interactive notebook environment (like Jupyter, but reactive).

```bash
# Run the single text analysis notebook
uv run marimo run marimo/count_verbs_single.py

# Or run the comparison notebook
uv run marimo run marimo/count_verbs_compare.py
```

This will open a browser window at `http://localhost:2718` with the interactive notebook.




## Development

To work on the code:

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Edit notebooks interactively
uv run marimo edit marimo/count_verbs_single.py
```

## Docker Build

```
docker build . --load -t correlaid/cdl-insl
```