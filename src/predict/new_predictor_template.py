from predict.predictor import Predictor

import sys
sys.path.append('..')
from prediction import Prediction

import numpy as np


class Newclass(Predictor):
    def __init__(self):
        # How many sample trajectories are predicted for each pedestrian
        self.num_samples = 5

    def predict(self, scenario):
        # Access the observation length and prediction horizon of the input scenario
        prediction_horizon = scenario.prediction_horizon
        observation_length = scenario.observation_legnth

        # Get the number of pedestrians in this scenario
        num_pedestrians = len(scenario.trajectories)

        # This is the container for the predicted sample trajectories
        prediction = []

        # Access the observed trajectories in this scenario
        for i in range(num_pedestrians):  # trajectory of person i
            trajectory = scenario.trajectories[i][:, :]

        for s in range(self.num_samples):
            # prediction_sample should contain one trajectory for each pedestrian
            prediction_sample = np.zeros((prediction_horizon,num_pedestrians,2))

            prediction.append(prediction_sample)

        return Prediction(prediction)