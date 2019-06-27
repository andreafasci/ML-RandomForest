# Machine Learning course - Random Forest implementation

Project for the course of Machine Learning - A. Y. 2018/19
In the project I will implement the Random Forest algorithm for classification purposes, using Decision Tree already implemented (from Sklearn library).
Code is available for both Python and Anaconda Jupyter.
I will use the dataset available at https://archive.ics.uci.edu/ml/datasets/Heart+Disease (in particular the file processed.cleveland.data) to test it and compare precisions with the Random Forest of the Sklearn library. 
That's why there are two versions of the codes
- *ML_Sklearn_Implementation* contains the preprocessing of data and the usage of Random Forest from the Sklearn library.
- *ML_My_Implementation* contains the preprocessing of data, the definition of my class for Random Forest and the usage of my implementation.

### Prerequisites

To run the Python version of the code, you will need Python 3.6. 
For Linux/Mac machines, you will probably already have Python, just check that your current version is the last available, running 
```
python --version
```
in your terminal, update in case you need it.
For Windows machines, you will need to install Python from this website: https://www.python.org/downloads/

To run the Jupyter version of the code, you will need to install Anaconda so that you will have both Python and Jupyter on your machine.
For both Linux/Windows/Mac you can find Anaconda at this link: https://www.anaconda.com/distribution/

### Running on Jupyter

Once you have started Jupyter and it has opened in your browser, just navigate to the right directory where you have the files, and open the code.
To run the code, just click (on top-right of your window) on Kernel -> Restart and Run All.
On the bottom of the page, in the last cell, you will see that the code is running. 
At the end of the execution it will show the average accuracy of the algorithm (executed K times - see the code -) and the standard deviation of the accuracy.

### Running on Python

Open your terminal and run

```
python ML_My_Implementation.py
```
or

```
python ML_Sklearn_Implementation.py
```
At the end of the execution it will show the average accuracy of the algorithm (executed K times - see the code -) and the standard deviation of the accuracy.


## Author

* **Andrea Fasciglione** (https://github.com/andreafasci)
