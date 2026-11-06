from abc import ABC
from abc import abstractmethod
from collections.abc import Callable

from typing import List
from typing import Tuple

Vec = List[int]


class Optimizer(ABC):
    def __init__(
        self, _solution_dimension: int, _fitness_function: Callable[[Vec], int]
    ):
        self.solution_dimension = _solution_dimension
        self.fitness_function = _fitness_function

    @abstractmethod
    def get_best(self, _xb: Vec, _fb: float) -> Tuple[Vec, float]:
        pass
