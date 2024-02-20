from typing import Callable
from typing import Literal
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def exponential_decay(
    initial_value: float, iteration: int, max_iterations: int
) -> float:
    return initial_value * np.exp(-iteration / max_iterations)


def linear_decay(initial_value: float, iteration: int, max_iterations: int) -> float:
    return initial_value * (1 - iteration / max_iterations)


DECAY_FUNCTIONS = {
    "exponential": exponential_decay,
    "linear": linear_decay,
}


class SelfOrganizingMap:
    """
    A simple implementation of a Self-Organizing Map (SOM).

    Attributes:
        dimensions (Tuple[int, int]): The dimensions of the SOM grid.
        learning_rate (float): The initial learning rate.
        max_iterations (int): The maximum number of iterations to train the SOM.
        radius (float): The initial radius of the neighborhood function.
        radius_decay_function (Literal["exponential", "linear"]):
        The function to decay the radius.
        learning_rate_decay_function (Literal["exponential", "linear"]):
        The function to decay the learning rate.
    """

    def __init__(
        self,
        dimensions: Tuple[int, int],
        learning_rate: float = 0.5,
        max_iterations: int = 100,
        radius: float | Literal["default"] = "default",
        radius_decay_function: Literal["exponential", "linear"] = "exponential",
        learning_rate_decay_function: Literal["exponential", "linear"] = "exponential",
        seed: int = 42,
    ) -> None:
        self._dimensions = dimensions
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._seed = seed
        self._rng = self._get_rng()
        self._radius = radius
        self._radius_decay_function = radius_decay_function
        self._learning_rate_decay_function = learning_rate_decay_function

    def _get_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._seed)

    @property
    def radius(self) -> float:
        if self._radius == "default":
            return max(self._dimensions) / 2
        return float(self._radius)

    @property
    def radius_decay_function(self) -> Callable[[float, int, int], float]:
        return DECAY_FUNCTIONS[self._radius_decay_function]

    @property
    def learning_rate_decay_function(self) -> Callable[[float, int, int], float]:
        return DECAY_FUNCTIONS[self._learning_rate_decay_function]

    def _find_bmu(self, X: NDArray) -> NDArray:
        """
        Finds the Best Matching Unit (BMU) for a given input vector.

        Parameters:
            input_vector (NDArray): The input vector to find the BMU for.

        Returns:
            NDArray: The position of the BMU.
        """
        return np.array(
            np.unravel_index(
                np.argmin(np.linalg.norm(self.map_ - X, axis=-1)), self.map_.shape[:-1]
            )
        )

    def _xy_distance(self, bmu_idx: NDArray) -> NDArray:
        xxyy = np.meshgrid(np.arange(self.map_.shape[0]), np.arange(self.map_.shape[1]))
        return np.linalg.norm(np.array(xxyy) - np.array(bmu_idx)[:, None, None], axis=0)

    def _radius_mask(self, xxyy_dist: NDArray, iteration: int) -> NDArray:
        return xxyy_dist <= self.radius_decay_function(
            self.radius, iteration, self._max_iterations
        )

    def _update_weights(
        self, input_vector: NDArray, bmu_idx: NDArray, iteration: int
    ) -> None:
        """
        Updates the weights of the SOM.

        Parameters:
            input_vector (NDArray): The input vector used for updating.
            bmu_idx (NDArray): The index of the Best Matching Unit.
            iteration (int): The current iteration number.
        """
        xxyy_dist = self._xy_distance(bmu_idx)
        radius_mask = self._radius_mask(xxyy_dist, iteration).T[:, :, None]

        self.map_ += (
            self.learning_rate_decay_function(
                self._learning_rate, iteration, self._max_iterations
            )
            * radius_mask
            * (input_vector - self.map_)
            * np.exp(-(xxyy_dist**2) / (2 * self.radius**2))[:, :, None]
        )

    def fit(self, X_train: NDArray) -> None:
        """
        Trains the SOM with the provided data.

        Parameters:
            X_train (NDArray): The input data to train the SOM with.
        """
        n_data, n_features = X_train.shape
        self.map_ = self._rng.random(
            (self._dimensions[0], self._dimensions[1], n_features)
        )

        for i in range(self._max_iterations):
            input_vector = X_train[self._rng.integers(0, n_data)]
            bmu_idx = self._find_bmu(input_vector)
            self._update_weights(input_vector, bmu_idx, i)

    def transform(self, X: NDArray) -> NDArray:
        """
        Maps the input data to the SOM grid.

        Parameters:
            X (NDArray): The input data to map.

        Returns:
            NDArray: The mapped input data.
        """
        return np.array([[self._find_bmu(input_vector) for input_vector in X]])

    def fit_transform(self, X: NDArray) -> NDArray:
        """
        Fits the SOM to the input data and returns the mapped data.

        Parameters:
            X (NDArray): The input data to fit the SOM to.

        Returns:
            NDArray: The mapped input data.
        """
        self.fit(X)
        return self.transform(X)
