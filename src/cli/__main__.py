"""
Pythia CLI Entry Point

Simple wrapper to allow: pythia [command]
"""

import sys
import os

# When installed, the project root is the parent of src/
# When running from source, it's the current directory
possible_roots = [
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # Installed: src/cli/ -> src/ -> project root
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),                     # Source: src/cli/ -> src/
    os.path.abspath(os.path.curdir),                                                  # Fallback: CWD
]

for root in possible_roots:
    main_file = os.path.join(root, 'main.py')
    if os.path.exists(main_file):
        if root not in sys.path:
            sys.path.insert(0, root)
        break

# Import main from main.py in project root
try:
    from main import main
except ImportError:
    import main as _main
    main = _main.main

if __name__ == '__main__':
    sys.exit(main())
