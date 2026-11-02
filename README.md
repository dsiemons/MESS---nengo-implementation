Neuromorphic Computing for the Resource-Constrained Project Scheduling Problem
This repository contains the Python scripts for generating RCPSP instances, running the neuromorphic experiments, and analysing their outputs.
Requirements
- Python 3.10 or later (adjust this version if your environment uses a different supported version)
- Nengo
- Matplotlib
- NumPy
- Pandas
- SciPy
python -m pip install --upgrade pip
python -m pip install nengo matplotlib numpy pandas scipy

It is recommended to run the project in a dedicated virtual environment:
- python -m venv .venv
Activate the environment:
Windows:
- .venv\\Scripts\\Activate.ps1
  macOS/Linux:
- source .venv/bin/activate

The scripts should be run in the correct order:
1. SGS.py
2. Instance_generation.py
3. batchRCPSP_Experiments.py
4. analysis.py
5. ClarityofOutput_test.py
6. test_BG_input.py
7. violations.py

You can run "run_all.py" to ensure the scripts are run in the right order. 
