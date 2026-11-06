from typing import Tuple


from numpy.random import randint
from numpy.random import uniform
from numpy.random import shuffle
from numpy import arange
from numpy import empty
from numpy import full


from . import Optimizer
from . import Vec


class OptimizerDifferentialEvolution(Optimizer):
    def __init__(
        self,
        _population_size: int,
        _scaling_factor: float,
        _crossover_rate: float,
        _orde: int,
        _use_best: bool,
        *args,
        **kwargs,
    ):
        if (_orde * 2 + 1) >= _population_size:
            raise AttributeError("Orde is too high for the current population size.")
        if not (0.0 < _scaling_factor < 2.0) or _scaling_factor == 1.0:
            raise AttributeError("Scaling factor should be in [0, 2)\\{1}")
        if not (0.0 < _crossover_rate < 1.0):
            raise AttributeError("Crossover rate should be in (0, 1)")
        if _population_size < 4:
            raise AttributeError("The population size must be >= 4.")

        super().__init__(*args, **kwargs)
        self.population_size = _population_size
        self.scaling_factor = _scaling_factor
        self.crossover_rate = _crossover_rate
        self.orde = _orde
        self.use_best = _use_best
        self.population = uniform(
            -1, 1, size=(_population_size, self.solution_dimension)
        )
        self.permutations = arange(_population_size)
        self.offspring = empty((_population_size, self.solution_dimension))
        self.fitness_values = full(_population_size, float("-inf"))

    def mutate(self, _xb: Vec):
        shuffle(self.permutations)
        for i in range(self.population_size):
            self.offspring[i] = (
                _xb[i] if self.use_best else self.population[self.permutations[i]]
            )
            for j in range(self.orde):
                x2 = self.population[
                    self.permutations[(i + 1 + j * 2) % self.population_size]
                ]
                x3 = self.population[
                    self.permutations[(i + 2 + j * 2) % self.population_size]
                ]
                self.offspring[i] += (x2 - x3) * self.scaling_factor

    def selection(self, _xb: Vec, _fb: float) -> Tuple[Vec, float]:
        for i in range(self.population_size):
            if self.fitness_values[i] <= (
                f := self.fitness_function(self.offspring[i])
            ):
                self.fitness_values[i] = f
                self.population[i] = self.offspring[i]
                if _fb <= f:
                    _fb = f
                    _xb = self.offspring[i].copy()
        return (_xb, _fb)

    def crossover(self):
        for i in range(self.population_size):
            random_j = randint(self.solution_dimension)
            for j in range(self.solution_dimension):
                if not (uniform(0, 1) <= self.crossover_rate or random_j == j):
                    self.offspring[i][j] = self.population[i][j]

    def get_best(self, _xb: Vec, _fb: float) -> Tuple[Vec, float]:
        self.mutate(_xb)
        self.crossover()
        return self.selection(_xb, _fb)
