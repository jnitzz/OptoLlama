#!/bin/bash
# Job script for running test_notebook.py with a results file.

# Activate the virtual environment
source E:/repos/OptoLlama/.venv/optollama/Scripts/Activate.ps1

# Run the test_notebook.py script with the results file
# Example usage: ./jobscripts/test_notebook.sh --config configs/optollama.yaml --results results.json
python scripts/test_notebook.py "$@"
