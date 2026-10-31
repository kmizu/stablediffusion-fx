"""Unambiguous remote references supplied by the MCP host.

Ordinary desktop model references retain their existing interpretation. An
explicit hf:// reference resolves through the Hub cache, so a worker-relative
directory with the same owner/repository name cannot shadow the remote model.
"""
import re


def encoder_source(value: str) -> str:
    if not value.startswith("hf://"):
        return value
    repo = value[5:]
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", repo):
        raise ValueError("Invalid explicit Hugging Face encoder reference")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo)
