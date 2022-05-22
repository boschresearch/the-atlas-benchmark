class Scenario:
    def __init__(self,startframe,scenario_trajectories, scenario_pedestrian_ids, scenario_gt, prediction_horizon, observation_len):
        self.startframe = startframe
        self.trajectories = scenario_trajectories
        self.pedestrian_ids = scenario_pedestrian_ids
        self.gt = scenario_gt

        self.observation_legnth = observation_len
        self.prediction_horizon = prediction_horizon