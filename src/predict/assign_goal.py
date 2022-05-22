# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

# Auxilary function, only used here in get_goal
def get_direction(trajectory):
    direction = []
    for i in range(3):
        b = trajectory[-1 - i] - trajectory[-2 - i]
        direction.append(b)
    total_direction = np.sum(direction, axis=0)  # (3,2)
    n_direction = total_direction / (np.linalg.norm(total_direction)+0.000001)  # unit direction vector
    return n_direction

# Auxilary function, only used here in get_goal
def get_goal_direction(trajectory, goal):
    goal_direction = []
    for i in range(len(goal)):
        one_direction = goal[i] - trajectory[-1]
        one_direction = one_direction / np.linalg.norm(one_direction)
        goal_direction.append(one_direction)
    return goal_direction

def get_goal(ped_trajectory, goal):
    sign_goal = 100 * np.ones(len(ped_trajectory), dtype=int)  # ndarray (# of ped,)
    for i in range(len(ped_trajectory)):
        if len(ped_trajectory[i]) >= 4:  # if len(trajectory)<5, no goal calculated
            final_direction = get_direction(ped_trajectory[i])
            goal_direction = get_goal_direction(ped_trajectory[i], goal)
            cos_a = np.dot(goal_direction, final_direction)
            sign_goal[i] = np.argmax(cos_a)
    return sign_goal
