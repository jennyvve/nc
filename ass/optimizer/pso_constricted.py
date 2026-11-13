from typing import Tuple


from numpy.random import uniform
from numpy import full
from numpy import sqrt


from . import Optimizer
from . import Vec
from . import FitnessCallable


class OptimizerPSOConstricted(Optimizer):
    def __init__(
        self,
        _population_size: int,
        _phi_1: float,
        _phi_2: float,
        *args,
        **kwargs,
    ):
        if not (_phi_1 + _phi_2) > 4:
            raise AttributeError("Expected: phi_1 + phi_2 > 4.")

        super().__init__(*args, **kwargs)
        self.phi_1 = _phi_1
        self.phi_2 = _phi_2
        self.phi = _phi_1 + _phi_2
        self.chi = 2 / (self.phi - 2 + sqrt(self.phi**2 - 4 * self.phi))
        self.population_size = _population_size
        self.population_pos = uniform(
            -1, 1, size=(_population_size, self.solution_dimension)
        )
        self.population_vel = uniform(
            -0.5, 0.5, size=(_population_size, self.solution_dimension)
        )
        self.person_best_fit = full(_population_size, float("-inf"))
        self.person_best_pos = self.population_pos.copy()

    def update_fitness(
        self, _xb: Vec, _fb: float, _fitness_function: FitnessCallable
    ) -> Tuple[Vec, float]:
        for i in range(self.population_size):
            x = self.population_pos[i]
            f = _fitness_function(x)
            if f > self.person_best_fit[i]:
                self.person_best_pos[i] = x.copy()
                self.person_best_fit[i] = f
            if f > _fb:
                _xb = x.copy()
                _fb = f
        return _xb, _fb

    def update_pos(self, _xb: Vec, _fb: float) -> None:
        for i in range(self.population_size):
            x = self.population_pos[i]
            pb = self.person_best_pos[i]

            cognitive = (
                self.phi_1 * uniform(0, 1, size=self.solution_dimension) * (pb - x)
            )
            social = (
                self.phi_2 * uniform(0, 1, size=self.solution_dimension) * (_xb - x)
            )

            self.population_vel[i] += cognitive + social
            self.population_vel[i] *= self.chi
            self.population_pos[i] += self.population_vel[i]

    def get_best(
        self, _xb: Vec, _fb: float, _fitness_function: FitnessCallable
    ) -> Tuple[Vec, float]:
        _xb, _fb = self.update_fitness(_xb, _fb, _fitness_function)
        self.update_pos(_xb, _fb)
        return _xb, _fb
