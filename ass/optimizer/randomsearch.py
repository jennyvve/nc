from typing import Tuple

from numpy import random


from . import Optimizer
from . import Vec


class OptimizerRandomSearch(Optimizer):

    def get_best(self, _xb: Vec, _fb: float) -> Tuple[Vec, float]:
        x = random.uniform(-1, 1, size=self.solution_dimension)
        if _fb < (f := self.fitness_function(x)):
            return (x, f)
        return (_xb, _fb)
