# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0


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
