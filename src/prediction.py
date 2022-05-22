import numpy as np

class Prediction:
    def __init__(self,trajectories):
        # joint trajectories for all prdestrians have the format [sample1, sample2, sample3...]
        # each sample is an np.array with the following structure: [time][pedestrian][x,y]
        if(type(trajectories) is np.ndarray):
            # If there is only one sample, for compatibility converting it into a list with one element
            self.trajectories = [trajectories]
        else:
            self.trajectories = trajectories