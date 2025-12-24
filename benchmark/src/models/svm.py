from __future__ import annotations

import numpy as np
from ConfigSpace import CategoricalHyperparameter, Configuration, ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from sklearn import svm
from sklearn.model_selection import cross_val_score
from src.datasets.dataset import Dataset
from src.models.model import Model


class SVMModel(Model):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        kernel = CategoricalHyperparameter("kernel", ["linear", "poly", "rbf", "sigmoid"], default_value="poly")
        C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0, log=True)
        shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
        degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)
        coef = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)
        gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")
        gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8.0, default_value=1.0, log=True)

        # Then we create dependencies
        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
        use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])

        # Add hyperparameters and conditions to our configspace
        cs.add([kernel, C, shrinking, degree, coef, gamma, gamma_value])
        cs.add([use_degree, use_coef, use_gamma, use_gamma_value])

        return cs

    def train(self, config: Configuration, instance: str | None, seed: int) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        assert self.dataset is not None
        config_dict = config.get_dictionary()
        if "gamma" in config:
            config_dict["gamma"] = config_dict["gamma_value"] if config_dict["gamma"] == "value" else "auto"
            config_dict.pop("gamma_value", None)

        # Get instance
        if instance is not None:
            data, target = self.dataset.get_instance_data(instance)

            classifier = svm.SVC(**config_dict, random_state=seed)
            scores = cross_val_score(classifier, data, target, cv=5)  # type: ignore
            cost = 1 - np.mean(scores)
        else:
            classifier = svm.SVC(**config_dict, random_state=seed)
            classifier.fit(self.dataset.get_X(), self.dataset.get_Y())
            accuracy = classifier.score(self.dataset.get_X(), self.dataset.get_Y())
            cost = 1 - accuracy

        return cost
