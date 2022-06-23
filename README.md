# CrackDect

Expandable crack detection for composite materials.



![alt text](docs/source/images/overview_gif.gif)

This package provides crack detection algorithms for tunneling off axis cracks in
glass fiber reinforced materials.

Full paper: [CrackDect: Detecting crack densities in images of fiber-reinforced polymers](https://www.sciencedirect.com/science/article/pii/S2352711021001205)  
Full documentation: [https://crackdect.readthedocs.io/en/latest/](https://crackdect.readthedocs.io/en/latest)

If you use this package in publications, please cite the paper. 

In this package, crack detection algorithms based on the works of Glud et al. [1] and Bender et al. [2] are implemented.
This implementation is aimed to provide a modular "batteries included" package for
this crack detection algorithms as well as a framework to preprocess image series to suite the 
prerequisites of the different crack detection algorithms.

## Quick start

To install CrackDect, check at first the [prerequisites](#Prerequisites) of your python installation. 
Upon meeting all the criteria, the package can be installed with pip, or you can clone or download the repo. 
If the installed python version or certain necessary packages are not compatible we recommend the use
of virtual environments by virtualenv or Conda.

Installation:

```pip install crackdect```

## Prerequisites

This package is written and tested in Python 3.8. The following packages must be installed.

* [scikit-image](https://scikit-image.org/>) 0.18.1
* [numpy](https://numpy.org/) 1.18.5
* [scipy](https://www.scipy.org/) 1.6.0
* [matplotlib](https://matplotlib.org/) 3.3.4
* [sqlalchemy](https://www.sqlalchemy.org/) 1.3.23
* [numba](https://numba.pydata.org/) 0.52.0
* [psutil](https://psutil.readthedocs.io/en/latest/) 5.8.0

## Motivation
Most algorithms and methods for scientific research are implemented as in-house code and not accessible for other
researchers. Code rarely gets published and implementation details are often not included in papers presenting the
results of these algorithms. Our motivation is to provide transparent and modular code with high level functions
for crack detection in composite materials and the framework to efficiently apply it to experimental evaluations.

## Contributing

Clone the repository and add changes to it. Test the changes and make a pull request.

## Authors
* Matthias Drvoderic

## License

This project is licensed under the MIT License.

## References

[1] [J.A. Glud, J.M. Dulieu-Barton, O.T. Thomsen, L.C.T. Overgaard
Automated counting of off-axis tunnelling cracks using digital image processing
Compos. Sci. Technol., 125 (2016), pp. 80-89](https://www.sciencedirect.com/science/article/abs/pii/S0266353816300197?via%3Dihub)

[2] [Bender JJ, Bak BLV, Jensen SM, Lindgaard E. 
Effect of variable amplitude block loading on intralaminar crack initiation and propagation in multidirectional GFRP laminate
Composites Part B: Engineering. 2021 Jul](https://www.researchgate.net/publication/350967596_Effect_of_variable_amplitude_block_loading_on_intralaminar_crack_initiation_and_propagation_in_multidirectional_GFRP_laminate)
