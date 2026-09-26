"""`uv run ipc-app` - launches the Streamlit UI (main.py) with any extra streamlit args."""

import sys

from ipc_rag.config import ROOT


def main() -> None:
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(ROOT / "main.py"), *sys.argv[1:]]
    sys.exit(cli.main())
