# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0


class Scenario:
    def __init__(self,startframe,scenario_trajectories, scenario_pedestrian_ids, scenario_gt, prediction_horizon, observation_len):
        self.startframe = startframe
        self.trajectories = scenario_trajectories
        self.pedestrian_ids = scenario_pedestrian_ids
        self.gt = scenario_gt

        self.observation_length = observation_len
        self.prediction_horizon = prediction_horizon
