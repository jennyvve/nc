from problem import GymProblem

from matplotlib.pyplot import subplots

from numpy import empty
from numpy import average
from numpy import arange

from tqdm import tqdm

from typing import Tuple

from . import Optimizer
from . import Vec


class OptimizerProducer(object):
    def __init__(self, _optimizer: Optimizer):
        self.optimizer = _optimizer
        self.plt_fitness_data = None

    def optimize(
        self, iteration_limit: int, termination_criteria: int
    ) -> Tuple[Vec, float]:
        self.plt_fitness_data = empty(iteration_limit, dtype=float)
        xb: Vec = None
        fb: float = float("-inf")

        print("Finding optimal solution.")
        for iteration in (pbar := tqdm(range(0, iteration_limit))):
            xb, fb = self.optimizer.get_best(xb, fb)
            self.plt_fitness_data[iteration] = fb
            pbar.set_postfix(best_fitness=fb)
            if fb >= termination_criteria:
                break

        return (xb, fb)

    def plt_fitness(self) -> None:
        if self.plt_fitness_data is None:
            print("Unable to plot fitness data: no fitness data available.")
            return None

        fig, ax = subplots()
        ax.plot(arange(len(self.plt_fitness_data)), self.plt_fitness_data)
        ax.set_xlabel("iteration")
        ax.set_ylabel("best fitness value")
        ax.set_title("Fitness value over time.")

    def get_fitness_average(self) -> int:
        return average(self.plt_fitness_data) if self.plt_fitness_data else 0
