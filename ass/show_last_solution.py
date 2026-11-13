from problem import GymProblem


from json import load

from numpy import array

if __name__ == "__main__":
    with open("solutions.json", "r", encoding="utf-8") as f:
        data = load(f)

    last = data["items"][-1]
    xb = array(last["solution"])
    use_lidar = last["use_lidar"]
    n_lidar = last["n_lidar"]
    pipe_gap = last["pipe_gap"]

    problem = GymProblem(use_lidar=use_lidar, n_lidar=n_lidar, pipe_gap=pipe_gap)

    problem.show_human(xb)
