from problem import GymProblem

from matplotlib.pyplot import subplots

from numpy import empty
from numpy import average
from numpy import arange

from tqdm import tqdm

from typing import Tuple

from json import dump
from json import load


from . import Optimizer
from . import Vec
from . import FitnessCallable


class OptimizerProducer(object):
    def __init__(self, _optimizer: Optimizer, _fitness_function: FitnessCallable):
        self.optimizer = _optimizer
        self.plt_fitness_data = None
        self.fitness_function = _fitness_function

    def optimize(
        self, iteration_limit: int, termination_criteria: int
    ) -> Tuple[Vec, float]:
        fitness_evals = 0

        def fitness_function_wrapper(*args, **kwargs) -> int:
            nonlocal fitness_evals
            fitness_evals += 1
            return self.fitness_function(*args, **kwargs)

        self.plt_fitness_data = empty(iteration_limit, dtype=float)
        xb: Vec = None
        fb: float = float("-inf")

        print("Finding optimal solution.")
        for iteration in (pbar := tqdm(range(0, iteration_limit))):
            xb, fb = self.optimizer.get_best(xb, fb, fitness_function_wrapper)
            self.plt_fitness_data[iteration] = fb
            pbar.set_postfix(best_fitness=fb, evals=fitness_evals)
            if fb >= termination_criteria:
                break

        return (xb, fb, fitness_evals)

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

    def store_solution(self, _xb: Vec, _fb: float, _fitness_evals: int, **kwargs):
        try:
            with open("solutions.json", "r", encoding="utf-8") as f:
                data = load(f)
                if "items" not in data:
                    raise Exception
        except:
            data = {"items": []}

        data["items"].append(
            {
                "fitness": _fb,
                "fitness_evals": _fitness_evals,
                "solution": list(_xb),
                **kwargs,
            }
        )

        with open("solutions.json", "w", encoding="utf-8") as f:
            dump(data, f)
