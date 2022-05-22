import numpy as np
import scipy.signal

import sys
sys.path.append('..')
from prediction import Prediction


class Predictor(object):
    def __init__(self, dataset):
        self.frame_frequency = dataset.frame_frequency
        self.world_d2 = dataset.map
        self.goal_list = dataset.goals
        self.delta_T = 1 / self.frame_frequency

class Predictor_CVM(Predictor):
    def __init__(self, dataset, method_cfg):
        Predictor.__init__(self, dataset)
        self.v0_mode = method_cfg['param']['meta']['v0_mode']
        self.load_goal_file = method_cfg['param']['meta']['load_goal']
        self.goal_step = method_cfg['param']['meta']['goal_step']
        self.v0_sigma = method_cfg['param']['meta']['v0_sigma']
        self.uncertainty = method_cfg['param']['uncertainty']['uncertainty']
        self.noise_scale = method_cfg['param']['uncertainty']['uncertainty sigma']
        self.num_particles = method_cfg['param']['uncertainty']['num_particles']

        self.consider_goals = method_cfg['param']['meta']['load_goal']
        self.obstacle_type = dataset.obstacle_type

    # this is the main function, which is used to generate predictions;
    # it should be present in every class, which inherits from Predictor
    def predict(self, scenario):
        if(not self.uncertainty):
            return Prediction([self.predict_single(scenario)])
        else:
            prediction = []
            for _ in range(self.num_particles):
                prediction.append(self.predict_single(scenario))
            return Prediction(prediction)

    def predict_single(self, scenario):
        prediction = []
        position = []
        v_mean = []
        # len(frame_tra[:] is the # of ped in this frame period
        # if (len(frame_tra[:]) >= 1):  # choose trajectory of ped[i] in this frame(most is [])
        for i in range(len(scenario.trajectories)):  # predict for i-ped in this frame
            old_position = scenario.trajectories[i][-1, :]
            mean_velocity = self.average_velocity(scenario.trajectories[i])
            position.append(old_position)
            v_mean.append(mean_velocity)
        v_mean = np.squeeze(v_mean)
        old_position = position
        for _ in range(scenario.prediction_horizon):
            new_position = old_position + v_mean * self.delta_T
            if self.uncertainty:
                noise = np.random.normal(0, self.noise_scale / self.frame_frequency, (len(new_position), 2))
                new_position = new_position + noise
            prediction.append(new_position)
            old_position = new_position
        return np.array(prediction)  # (prediction_horizon, # of ped, 2)

    # here, the assumption of frame_tra: (len(frame_tra[i]) >= 2)  and (len(frame_tra) != 0)
    def average_velocity(self, trajectory):
        velocity = (trajectory[1::, :] - trajectory[0:-1, :]) / self.delta_T
        w = np.expand_dims(self.get_w(velocity), axis=0)
        new_velocity = np.dot(w, velocity)
        return new_velocity

    def get_w(self, velocity):  # smaller sigma is like CVM (last step projection), bigger sigma is like LIN (uniform velocity filtering)
        velocity_len = len(velocity)
        if self.v0_mode == 'linear':
            w = np.ones(velocity_len) * 1 / velocity_len
        if self.v0_mode == 'gaussian':
            window = scipy.signal.gaussian(2 * velocity_len, self.v0_sigma)
            w1 = window[0:velocity_len]
            scale = np.sum(w1)
            # scale = np.linalg.norm(w1)
            w = w1 / scale
            # print(w)
        if self.v0_mode == 'constant':
            w = np.zeros(velocity_len)
            w[-1] = 1
        return w

    def get_goal_short(self, frame_tra):
        position = []  # the position of last history timestep, if 2 trajectories, 2 lists, each list is ndarray (2,)
        v_mean = []
        new_position = []
        # if (len(frame_tra[:]) >= 1):  # choose trajectory of ped[i] in this frame(most is [])
        for i in range(len(frame_tra)):
            old_position = frame_tra[i][-1, :]
            mean_velocity = self.average_velocity(frame_tra[i])
            position.append(old_position)
            v_mean.append(mean_velocity)
        v_mean = np.squeeze(v_mean)
        old_position = position
        #for _ in range(self.goal_step):
        #    new_position = old_position + v_mean * self.delta_T
        #    old_position = new_position
        new_position = old_position + self.goal_step * v_mean * self.delta_T
        return new_position