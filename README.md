# CS 170 Project Spring 2021

Take a look at the project spec before you get started!

Requirements:

Python 3.6+

You'll only need to install networkx to work with the starter code. For installation instructions, follow: https://networkx.github.io/documentation/stable/install.html

If using pip to download, run `python3 -m pip install networkx`


Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions


How to generate outputs:
- Three different 'solve' functions to generate outputs are stored in solver_ver1.py, solver_ver2.py, and solver_ver3.py. 
- You can copy and paste each 'solve' function to solver.py and run 'python3 solver.py'
- Output directory is initially set to 'tempOutputs' directory. 
- You may modify line 169 in solver.py to change output directory accordingly.
- Optionally, you can use 'compare_scores' and 'move_better_outputs' functions in utils.py to compare and store different outputs generated from same input.

How I generated outputs:
- I first generated outputs to 'outputs' directory using 'solve' function in solver_ver1.py with HTC=3 for small, medium and HTC=2 for large.
- Then I generated outputs to 'tempOutputs' directory using 'solve' function in solver_ver2.py with HTC=1-5 for small, HTC=1-4 for medium, and HTC=1-3 for large. 
- Then I generated outputs to 'tempOutputs' directory using 'solve' function in solver_ver3.py with HTC_V=1-6 HTC_H=10 for small, HTC_V=1-5 HTC_H=5 for medium, and HTC_V=4 HTC_H=3 for large.

- I ran 'python3 solver.py' to generate outputs, and I used 'move_better_outputs' function to only store better results from 'tempOutputs' to 'outputs' directory. Then I deleted the outputs generated in 'tempOutputs' directory. 
