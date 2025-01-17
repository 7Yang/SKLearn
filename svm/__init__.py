"""
The :mod:`sklearn.svm` module includes Support Vector Machine algorithms.
"""

# See http://scikit-learn.sourceforge.net/modules/svm.html for complete documentation.

# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr> with help from
#         the scikit-learn community. LibSVM and LibLinear are copyright
#         of their respective owners.
# License: BSD 3 clause (C) INRIA 2010

from ._bounds import l1_min_c
from ._classes import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM

__all__ = [ "LinearSVC", "LinearSVR", "NuSVC", "NuSVR", "OneClassSVM", "SVC", "SVR", "l1_min_c" ]
