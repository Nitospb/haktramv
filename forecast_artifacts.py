"""Resolve each experiment independently: a fresh run overrides its snapshot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def artifact_directory(family):
    current = ROOT / 'outputs' / family
    return current if current.is_dir() else ROOT / 'outputs/studio' / family
