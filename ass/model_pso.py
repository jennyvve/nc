from problem import GymProblem


from matplotlib.pyplot import show
from matplotlib.pyplot import savefig


from optimizer import OptimizerProducer
from optimizer import OptimizerPSOConstricted

if __name__ == "__main__":
    PIPE_GAP = 120
    USE_LIDAR = True
    N_LIDAR = 16
    SOLUTION_DIMENSION = N_LIDAR * 2
    POPULATION_SIZE = 32
    PHI_1 = 2.05
    PHI_2 = 2.05
    FITNESS_LIMIT = 300
    ITERATION_LIMIT = 30
    problem = GymProblem(
        use_lidar=USE_LIDAR,
        n_lidar=N_LIDAR,
        pipe_gap=PIPE_GAP,
        reward_limit=FITNESS_LIMIT,
    )
    optimizer = OptimizerPSOConstricted(
        POPULATION_SIZE,
        PHI_1,
        PHI_2,
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
        phi_1=PHI_1,
        phi_2=PHI_2,
    )

    print(f"The best solution found has fitness value: {fb}")
    savefig("solution.png")
