from problem import GymProblem

from json import dump

from matplotlib.pyplot import show
from matplotlib.pyplot import savefig


from optimizer import OptimizerProducer
from optimizer import OptimizerRandomSearch
from optimizer import OptimizerDifferentialEvolution

if __name__ == "__main__":
    PIPE_GAP = 120
    USE_LIDAR = True
    N_LIDAR = 180
    SOLUTION_DIMENSION = N_LIDAR * 2
    POPULATION_SIZE = 8
    SCALING_FACTOR = 0.5
    CROSSOVER_RATE = 0.7
    ORDE = 1
    USE_BEST = False
    problem = GymProblem(use_lidar=USE_LIDAR, n_lidar=N_LIDAR, pipe_gap=120)
    optimizer = OptimizerDifferentialEvolution(
        POPULATION_SIZE,
        SCALING_FACTOR,
        CROSSOVER_RATE,
        ORDE,
        USE_BEST,
        SOLUTION_DIMENSION,
        problem.show_machine,
    )
    producer = OptimizerProducer(optimizer)

    xb, fb = producer.optimize(50, 500)
    producer.plt_fitness()

    print(f"The best solution found has fitness value: {fb}")
    savefig("solution.png")

    with open("last_solution.json", "w", encoding="utf-8") as f:
        dump(
            {
                "fitness": fb,
                "solution": list(xb),
                "use_lidar": USE_LIDAR,
                "n_lidar": N_LIDAR,
                "pipe_gap": PIPE_GAP,
            },
            f,
        )
