"""pytest configuration for the agent-loop-basic problem.

Adds the problem root to ``sys.path`` so tests can ``from solution import ...``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
