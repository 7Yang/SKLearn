"""
The :mod:`sklearn.metrics` module includes score functions, performance metrics and pairwise metrics and distance computations.
"""


from . import cluster
from ._dist_metrics import DistanceMetric
from ._plot.confusion_matrix import ConfusionMatrixDisplay
from ._plot.det_curve import DetCurveDisplay
from ._plot.precision_recall_curve import PrecisionRecallDisplay
from ._plot.regression import PredictionErrorDisplay
from ._plot.roc_curve import RocCurveDisplay

from ._classification import (accuracy_score, balanced_accuracy_score, brier_score_loss, class_likelihood_ratios,
    classification_report, cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, hamming_loss, hinge_loss,
    jaccard_score, log_loss, matthews_corrcoef, multilabel_confusion_matrix, precision_recall_fscore_support, precision_score,
    recall_score, zero_one_loss)

from ._ranking import (auc, average_precision_score, coverage_error, dcg_score, det_curve, label_ranking_average_precision_score, 
    label_ranking_loss, ndcg_score, precision_recall_curve, roc_auc_score, roc_curve, top_k_accuracy_score)

from ._regression import (d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, explained_variance_score, max_error, mean_absolute_error,
    mean_absolute_percentage_error, mean_gamma_deviance, mean_pinball_loss, mean_poisson_deviance, mean_squared_error, mean_squared_log_error,
    mean_tweedie_deviance, median_absolute_error, r2_score, root_mean_squared_error, root_mean_squared_log_error)

from ._scorer import check_scoring, get_scorer, get_scorer_names, make_scorer

from .cluster import (adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, completeness_score, consensus_score,
    davies_bouldin_score, fowlkes_mallows_score, homogeneity_completeness_v_measure, homogeneity_score, mutual_info_score,
    normalized_mutual_info_score, pair_confusion_matrix, rand_score, silhouette_samples, silhouette_score, v_measure_score)

from .pairwise import (euclidean_distances, nan_euclidean_distances, pairwise_distances, pairwise_distances_argmin, 
    pairwise_distances_argmin_min, pairwise_distances_chunked, pairwise_kernels)

__all__ = [
    "accuracy_score", "adjusted_mutual_info_score", "adjusted_rand_score", "auc", "average_precision_score", "balanced_accuracy_score",
    "calinski_harabasz_score", "check_scoring", "class_likelihood_ratios", "classification_report", "cluster", "cohen_kappa_score",
    "completeness_score", "ConfusionMatrixDisplay", "confusion_matrix", "consensus_score", "coverage_error", "d2_tweedie_score",
    "d2_absolute_error_score", "d2_pinball_score", "dcg_score", "davies_bouldin_score", "DetCurveDisplay", "det_curve",
    "DistanceMetric", "euclidean_distances", "explained_variance_score", "f1_score", "fbeta_score", "fowlkes_mallows_score",
    "get_scorer", "hamming_loss", "hinge_loss", "homogeneity_completeness_v_measure", "homogeneity_score", "jaccard_score",
    "label_ranking_average_precision_score", "label_ranking_loss", "log_loss", "make_scorer", "nan_euclidean_distances",
    "matthews_corrcoef", "max_error", "mean_absolute_error", "mean_squared_error", "mean_squared_log_error", "mean_pinball_loss",
    "mean_poisson_deviance", "mean_gamma_deviance", "mean_tweedie_deviance", "median_absolute_error", "mean_absolute_percentage_error",
    "multilabel_confusion_matrix", "mutual_info_score", "ndcg_score", "normalized_mutual_info_score", "pair_confusion_matrix",
    "pairwise_distances", "pairwise_distances_argmin", "pairwise_distances_argmin_min", "pairwise_distances_chunked", "pairwise_kernels",
    "PrecisionRecallDisplay", "precision_recall_curve", "precision_recall_fscore_support", "precision_score", "PredictionErrorDisplay",
    "r2_score", "rand_score", "recall_score", "RocCurveDisplay", "roc_auc_score", "roc_curve", "root_mean_squared_log_error",
    "root_mean_squared_error", "get_scorer_names", "silhouette_samples", "silhouette_score", "top_k_accuracy_score",
    "v_measure_score", "zero_one_loss", "brier_score_loss" ]
