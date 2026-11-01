"""Execute notebooks from clean kernels and retain outputs for review."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=300, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.name}", flush=True)
