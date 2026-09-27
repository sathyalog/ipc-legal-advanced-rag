"""IPC RAG package."""

import os
import shutil
from importlib.util import find_spec
from pathlib import Path


def _use_unlinked_nltk_data() -> None:
    """Point NLTK at a plain copy of llama-index's bundled NLTK data.

    On Linux, uv installs packages as hardlinks into its cache, and NLTK >= 3.10's
    pathsec refuses to open any file with st_nlink > 1 (PermissionError on
    Streamlit Cloud). Copying creates fresh single-link files.
    """
    if "NLTK_DATA" in os.environ:
        return
    spec = find_spec("llama_index.core")
    if not spec or not spec.submodule_search_locations:
        return
    bundled = Path(spec.submodule_search_locations[0]) / "_static" / "nltk_cache"
    if not bundled.is_dir():
        return
    target = Path(__file__).resolve().parent.parent / ".cache" / "nltk_data"
    if not (target / "corpora" / "stopwords").is_dir():
        shutil.copytree(bundled, target, dirs_exist_ok=True)
    os.environ["NLTK_DATA"] = str(target)


_use_unlinked_nltk_data()
