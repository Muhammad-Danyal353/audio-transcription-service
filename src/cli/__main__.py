"""
CLI entry point for the audio transcription service.

This module provides the main entry point for running the CLI as a module:
python -m src.cli
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.advanced_client import main

if __name__ == "__main__":
    main()
