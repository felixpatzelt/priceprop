PriceProp
=========

Simulate and calibrate linear propagator models for price responses to an
external order flow. The models and methods are explained and applied to
real high-frequency trading data in:
    
    Patzelt, F. and Bouchaud, J-P. (2017):
    Nonlinear price impact from linear models. 
    Journal of Statistical Mechanics: Theory and Experiment, 12, 123404. 
    Preprint at `arXiv:1708.02411 <//arxiv.org/abs/1708.02411>`_.
    
=====================   ======================================================
Function                Synopsis
=====================   ======================================================
G_pow                   Return power law Propagator kernel
beta_from_gamma         Return exponent beta for a power law propagator kernel  
                        that decorrelates an input with a pure power law 
                        autocorrelation with exponent gamma
calibrate_hdim2         Calibrate two-kernel History Dependent Impact Model
calibrate_tim1          Calibrate original Transient Impact Model
calibrate_tim2          Calibrate two-kernel Transient Impact Model
hdim2                   Simulate two-kernel History Dependent Impact Model
integrate               Return lag 1 sum, i.e. convert a differential kernel
                        to a "bare response".
k_pow                   Return differential form of power law propagator kernel
propagate               Apply propagator kernel to a time series (FFT conv.)
response                Calculate e.g. a price response
response_grouped_df     Calculate response for pandas groups and average
smooth_tail_rbf         Smooth the tail of a long kernel using logarithmically
                        spaced Radial Basis Functions
tim1                    Simulate original Transient Impact Model
tim2                    Simulate two-kernel Transient Impact Model
=====================   ======================================================


The submodule ``batch`` automates model calibration and simulation. Please
find further explanations in the docstrings and in the examples directory.

The required methods to efficiently estimate two- and three-point 
correlation matrices were released in the separate package 
`scorr <//github.com/felixpatzelt/scorr>`_.


Installation
------------

    pip install priceprop
    
    
Dependencies (automatically installed)
--------------------------------------

    - Python 2.7
    - NumPy
    - SciPy
    - Pandas
    - scorr
    
    
Optional Dependencies required only for the examples (pip installable)
----------------------------------------------------------------------

    - Jupyter
    - Matplotlib
    - colorednoise