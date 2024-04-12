"""
The :mod:`sklearn.manifold` module implements data embedding techniques.
"""

from ._isomap import Isomap
from ._mds import MDS, smacof
from ._t_sne import TSNE, trustworthiness
from ._locally_linear import LocallyLinearEmbedding, locally_linear_embedding
from ._spectral_embedding import SpectralEmbedding, spectral_embedding

__all__ = [
    "locally_linear_embedding", "LocallyLinearEmbedding", "Isomap", "MDS", "smacof", "SpectralEmbedding", "spectral_embedding", "TSNE", "trustworthiness" 
]
