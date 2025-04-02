# BOC Curve Generator

This software generates BOC curves and its associated probability density functions.
It replicates two key figures from our scientific paper by generating Build-Out Curves (BOC) for urban growth analysis.

**Dataset**
The data folder contains information about urban growth in Morelia Municipality:

Dependent variable (2018-2021 urban gain): Morelia_validation_Y.csv

Independent variables (initial conditions at 2018): Morelia_validation_X.csv:

1. Travel time

2. Distance to roads

3. Distance to urban cover

4. Distance to protected areas

**Requirements**

Python 3.x

Jupyter Notebook

Required Python packages

pandas
numpy
matplotlib.pyplot
importlib


**Usage**
Open main_BOC-SOFTWARE.ipynb in Jupyter Notebook

Run all cells sequentially

The notebook will generate:

BOC curves, Likelihood/Marginal ratio and Probability density functions for all 4 variables (replicating Figure 7 from the manuscript)

Combination of two posterior probabilities of the different rank variables (replicating Figure 9 from the manuscript)

**Implementation Details**

The core analysis functions are implemented in boc_analysis.py, which contains:

Probability function construction

BOC curve generation

Posterior distribution calculations

**Output**
Running the notebook will produce visualizations matching:

Figure 7 

Figure 9 
