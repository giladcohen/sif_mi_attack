import logging
from typing import Optional, TYPE_CHECKING
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import Module

import numpy as np
from captum.influence import TracInCPFast

from research.utils import load_state_dict

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class TracInAttack(MembershipInferenceAttack):
    attack_params = MembershipInferenceAttack.attack_params + [
        "influence_score_threshold",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE", influence_score_threshold: Optional[float] = None):
        """
        Create a `LabelOnlyDecisionBoundary` instance for Label-Only Inference Attack based on Decision Boundary.

        :param estimator: A trained classification estimator.
        :param distance_threshold_tau: Threshold distance for decision boundary. Samples with boundary distances larger
                                       than threshold are considered members of the training dataset.
        """
        super().__init__(estimator=estimator)
        self.influence_score_threshold = influence_score_threshold
        self.device_influence = 'cpu'
        self.device = 'cuda'
        self.batch_size = 100
        self.threshold_bins: list = []
        self._check_params()
        self.influence_model = None

    def fit(self,
            x_member: np.ndarray,
            y_member: np.ndarray,
            x_non_member: np.ndarray,
            y_non_member: np.ndarray,
            checkpoint_path: str):
        if x_member.shape[0] != y_member.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x_member and y_member do not match")
        if x_non_member.shape[0] != y_non_member.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")

        logger.info('Fitting class ' + __name__)
        y_member_pred = self.estimator.predict(x_member, batch_size=self.batch_size).argmax(axis=1)
        y_non_member_pred = self.estimator.predict(x_non_member, batch_size=self.batch_size).argmax(axis=1)

        self.estimator.model.to(self.device_influence)
        member_src_set = TensorDataset(torch.from_numpy(x_member).to(self.device_influence),
                                       torch.from_numpy(y_member).to(self.device_influence))
        self.influence_model = TracInCPFast(
            model=self.estimator.model,
            final_fc_layer=self.estimator.model.linear,
            influence_src_dataset=member_src_set,
            checkpoints_load_func=load_state_dict,
            checkpoints=[checkpoint_path],
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            batch_size=self.batch_size,
            vectorize=False)

        member_proponents_indices, member_proponents_influence_scores = self.influence_model.influence(
            torch.from_numpy(x_member).to(self.device_influence), torch.from_numpy(y_member).to(self.device_influence),
            k=x_member.shape[0], proponents=True)
        non_member_proponents_indices, non_member_proponents_influence_scores = self.influence_model.influence(
            torch.from_numpy(x_non_member).to(self.device_influence), torch.from_numpy(y_non_member).to(self.device_influence),
            k=x_non_member.shape[0], proponents=True)
        scores_member = member_proponents_influence_scores.sum(axis=1).cpu().numpy()
        scores_non_member = non_member_proponents_influence_scores.sum(axis=1).cpu().numpy()

        min_thd = np.min([scores_member.min(), scores_non_member.min()])
        max_thd = np.max([scores_member.max(), scores_non_member.max()])

        scores_member[y_member_pred != y_member] = -np.inf
        scores_non_member[y_non_member_pred != y_non_member] = -np.inf

        num_increments = 100
        thd_increment = (max_thd - min_thd) / num_increments
        acc_max = 0.0
        influence_threshold = None
        self.threshold_bins = []
        for i in range(1, num_increments):
            # searching for threshold that yields the best accuracy in separating between members and non-members
            thd = min_thd + i * thd_increment
            member_is_member = np.where(scores_member > thd, 1, 0)
            non_member_is_member = np.where(scores_non_member > thd, 1, 0)
            acc = (np.sum(member_is_member) + (non_member_is_member.shape[0] - np.sum(non_member_is_member))) / (
                    member_is_member.shape[0] + non_member_is_member.shape[0]
            )
            self.threshold_bins.append((thd, acc))
            if acc > acc_max:
                influence_threshold = thd
                acc_max = acc

        self.influence_score_threshold = influence_threshold
        # return net to original device
        self.estimator.model.to(self.device)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")

        if y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")

        if self.influence_score_threshold is None:  # pragma: no cover
            raise ValueError(
                "No value for threshold `influence_score_threshold` provided. Please set"
                "`influence_score_threshold` or run method `fit` on known training set."
            )

        y_pred = self.estimator.predict(x, self.batch_size).argmax(axis=1)
        self.estimator.model.to(self.device_influence)
        proponents_indices, proponents_influence_scores = self.influence_model.influence(
            torch.from_numpy(x).to(self.device_influence), torch.from_numpy(y).to(self.device_influence),
            k=x.shape[0], proponents=True)
        scores = proponents_influence_scores.sum(axis=1).cpu().numpy()
        scores[y_pred != y] = -np.inf
        predicted_class = np.where(scores > self.influence_score_threshold, 1, 0)
        # return net to original device
        self.estimator.model.to(self.device)
        return predicted_class

    def _check_params(self) -> None:
        if self.influence_score_threshold is not None and (
                not isinstance(self.influence_score_threshold, (int, float)) or self.influence_score_threshold <= 0.0
        ):
            raise ValueError("The influence threshold `influence_score_threshold` needs to be a positive float.")
