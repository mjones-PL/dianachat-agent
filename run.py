#!/usr/bin/env python3
"""
Startup script for DianaChat Agent.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

if __name__ == "__main__":
    # Import and run the agent
    from dianachat_agent.main import main
    main()
