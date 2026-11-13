from problem import GymProblem

from json import dump

from matplotlib.pyplot import show
from matplotlib.pyplot import savefig


from optimizer import OptimizerProducer
from optimizer import OptimizerDE

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
    FITNESS_LIMIT = 300
    ITERATION_LIMIT = 30
    problem = GymProblem(
        use_lidar=USE_LIDAR,
        n_lidar=N_LIDAR,
        pipe_gap=PIPE_GAP,
        reward_limit=FITNESS_LIMIT,
    )
    optimizer = OptimizerDE(
        POPULATION_SIZE,
        SCALING_FACTOR,
        CROSSOVER_RATE,
        ORDE,
        USE_BEST,
        SOLUTION_DIMENSION,
    )
    producer = OptimizerProducer(optimizer, problem.show_machine)

    xb, fb, fitness_evals = producer.optimize(ITERATION_LIMIT, FITNESS_LIMIT)
    producer.plt_fitness()

    producer.store_solution(
        xb,
        fb,
        fitness_evals,
        fitness_limit=FITNESS_LIMIT,
        iteration_limit=ITERATION_LIMIT,
        use_lidar=USE_LIDAR,
        n_lidar=N_LIDAR,
        pipe_gap=PIPE_GAP,
        population_size=POPULATION_SIZE,
        scaling_factor=SCALING_FACTOR,
        crossover_rate=CROSSOVER_RATE,
        orde=ORDE,
        use_best=USE_BEST,
    )

    print(f"The best solution found has fitness value: {fb}")
    savefig("solution.png")
