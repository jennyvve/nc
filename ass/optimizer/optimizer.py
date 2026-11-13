from abc import ABC
from abc import abstractmethod
from collections.abc import Callable

from typing import List
from typing import Tuple

Vec = List[int]
FitnessCallable = Callable[[Vec], int]


class Optimizer(ABC):
    def __init__(
        self,
        _solution_dimension: int,
    ):
        self.solution_dimension = _solution_dimension

    @abstractmethod
    def get_best(
        self, _xb: Vec, _fb: float, _fitness_function: FitnessCallable
    ) -> Tuple[Vec, float]:
        pass
