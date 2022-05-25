import logging
from typing import Optional, TYPE_CHECKING
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import Module

import numpy as np

from research.utils import load_state_dict

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class InfluenceFunctionDiff(MembershipInferenceAttack):
    attack_params = MembershipInferenceAttack.attack_params + [
        "influence_thd",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE",
                 influence_thd: Optional[float] = None):
        super().__init__(estimator=estimator)
        self.influence_thd = influence_thd
        self.device = 'cuda'
        self.batch_size = 100
        self._check_params()

    # def fit(self, influences_member_train, influences_non_member_train):
    #     if influences_member_train.shape[0] != influences_non_member_train.shape[0]:  # pragma: no cover
    #         raise ValueError("Number of rows in influences_member_train and influences_non_member_train do not match")
    #
    #     minn = self_influences_member_train.min()
    #     maxx = self_influences_member_train.max()
    #     delta = maxx - minn
    #     if self.influence_score_min is None:
    #         self.influence_score_min = minn - delta * 0.03
    #     if self.influence_score_max is None:
    #         self.influence_score_max = maxx + delta * 0.03

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")

        influences = kwargs.pop('influences', None)
        assert y.shape[0] == x.shape[0] == influences.shape[0], "Number of rows in x and y do not match"

        if self.influence_thd is None:  # pragma: no cover
            raise ValueError(
                "No value for threshold `influence_thd` provided. Please set them"
                "or run method `fit` on known training set."
            )

        y_pred = self.estimator.predict(x, self.batch_size).argmax(axis=1)
        scores = influences
        predicted_class = np.ones(x.shape[0])  # member by default
        for i in range(x.shape[0]):
            if scores[i] > self.influence_thd:
                predicted_class[i] = 0
            if y_pred[i] != y[i]:
                predicted_class[i] = 0

        return predicted_class

    def _check_params(self) -> None:
        if self.influence_thd is not None and not isinstance(self.influence_thd, (int, float)):
            raise ValueError("The influence threshold `influence_thd` needs to be a float.")
