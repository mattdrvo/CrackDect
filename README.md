# CrackDect

Expandable crack detection for composite materials.

![alt text](docs/source/images/overview_gif.gif)

This package provides an automated crack detection for tunneling off axis cracks in glass fiber reinforced materials.
It relies on image processing and works with transilluminated white light images (TWLI). The basis of the
crack detection method was first published by Glud et al. [1]. This implementation is aimed to provide a modular
"batteries included" package for this method and extensions of it as well as image preprocessing functions.

## Quick start

To install CrackDect, check at first the [prerequisites](#Prerequisites) of your python installation. 
Upon meeting all the criteria, the package can be installed with pip, or you can clone or download the repo. 
If the installed python version or certain necessary packages are not compatible we recommend the use
of virtual environments by virtualenv or Conda.

Installation:

```pip install crackdect```

Documentation:

[https://crackdect.readthedocs.io/en/latest/](https://crackdect.readthedocs.io/en/latest)


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

[1] J.A. Glud, J.M. Dulieu-Barton, O.T. Thomsen, L.C.T. Overgaard
Automated counting of off-axis tunnelling cracks using digital image processing
Compos. Sci. Technol., 125 (2016), pp. 80-89

