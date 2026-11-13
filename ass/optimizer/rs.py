from typing import Tuple

from numpy import random


from . import Optimizer
from . import Vec
from . import FitnessCallable


class OptimizerRandomSearch(Optimizer):

    def get_best(
        self, _xb: Vec, _fb: float, _fitness_function: FitnessCallable
    ) -> Tuple[Vec, float]:
        x = random.uniform(-1, 1, size=self.solution_dimension)
        if _fb < (f := _fitness_function(x)):
            return (x, f)
        return (_xb, _fb)
