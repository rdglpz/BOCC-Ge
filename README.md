# BOC-SOFTWARE

This software generates the BOC curve for the databse in data folder.

The data folder includes the information of urban gain of Morelia Municipality that ocurs at the final of the interval 2018-2021, and the indepentent variables that represent the conditions at the initial of the time interval.

The indepentent variables in Morelia_validation_X.csv

1 Travel Time
2 Distance to Roads
2 Distance to urban cover,
3 Distance to protected areas.


The dependent variable urban gain is in Morelia_validation_Y.csv

In order to run the BOC software, you need to open and run the cells sequentially of the main_BOC-SOFTWARE.ipynb. 

This notebook generates the BOC curves and probability density functions the of 4 variables presented in the Figure 9 of the manuscript and the combination of the posteriors in figure 7.

The main_BOC-SOFTWARE.ipynb imports the boc_analysis.py containing the functions to constructo the probability functions involved in the BOC curve presented in the manuscript.





 
