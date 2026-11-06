from problem import GymProblem


from json import load

from numpy import array

if __name__ == "__main__":
    with open("last_solution.json", "r", encoding="utf-8") as f:
        data = load(f)

    xb = array(data["solution"])
    use_lidar = data["use_lidar"]
    n_lidar = data["n_lidar"]
    pipe_gap = data["pipe_gap"]

    problem = GymProblem(use_lidar=use_lidar, n_lidar=n_lidar, pipe_gap=pipe_gap)

    problem.show_human(xb)
