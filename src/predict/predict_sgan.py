# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0

# System imports
import os
import sys
import numpy as np
import pandas as pd
import torch
import json
import time
import copy
import argparse
import os
import torch
import sys
from attrdict import AttrDict

# Prediction 
from predict.predictor import Predictor
sys.path.append('..')
from prediction import Prediction


# Adding trajectronplusplus components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sgan')))

# Imports from sgan
# 
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default= '../s-gan/models/sgan-models/eth_12_model.pt', type=str)
parser.add_argument('--num_samples', default=25, type=int)
parser.add_argument('--dset_type', default='test', type=str)


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEBUG = False

class SGANPredictor(Predictor):
    '''
    This class is the predictor that outputs the predicted future human trajectories that are used in the controller
    Input:

    close_humans:
    A dictionary with all the observed N humans in the area. The keys will be numbers and the values 
    will be np.arrays with the state with [x, y, vx, vy].

    '''

    def __init__(self,
                 scenario,
                 dataset,
                 method_cfg={},
                 num_modes=25
    ):
        Predictor.__init__(self, dataset)
        self.dt = self.delta_T 
        self.num_modes = num_modes  # trajectron only outputs 25 modes
        self.prediction_horizon = scenario.prediction_horizon
        self.cnt_time_steps = 0
        self.max_scene_list_size = 400
        self.mean_x = 0
        self.mean_y = 0
        self.scenario = scenario
        self.model_path = '../sgan/models/sgan-models/eth_12_model.pt'
        self.num_samples = 25
        self.dset_type = 'test'
        self.obs_len = 8
        if(method_cfg):
            self.uncertainty = method_cfg['uncertainty']
            self.uncertainty_sigma = method_cfg['uncertainty sigma']
            self.num_particles = method_cfg['num_particles']
        else:
            self.uncertainty = False
            self.uncertainty_sigma = 0.1
            self.num_particles = 1
    
    
    def get_generator(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len, #<---- prediction length
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.cuda()
        generator.train()
        self.obs_len = args.obs_len
        return generator


    def evaluate_helper(self, error, seq_start_end):
        sum_ = 0
        error = torch.stack(error, dim=1)

        for (start, end) in seq_start_end:
            start = start.item()
            end = end.item()
            _error = error[start:end]
            _error = torch.sum(_error, dim=0)
            _error = torch.min(_error)
            sum_ += _error
        return sum_


    def evaluate(self, args, loader, generator, num_samples):
        total_traj = 0
        with torch.no_grad():
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end) = batch
        
                total_traj += pred_traj_gt.size(1)

                predictions = dict()
                for z in range(num_samples):
                    pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end
                    )
                    pred_traj_fake = relative_to_abs(
                        pred_traj_fake_rel, obs_traj[-1]
                    )
                    predictions[z] = pred_traj_fake

                return predictions


    def predictSGAN(self, close_humans):
        if os.path.isdir(self.model_path):
            filenames = os.listdir(self.model_path)
            filenames.sort()
            paths = [
                os.path.join(self.model_path, file_) for file_ in filenames
            ]
        else:
            paths = [self.model_path]
        for path in paths:
            checkpoint = torch.load(path, map_location = 'cuda')
            generator = self.get_generator(checkpoint)
            _args = AttrDict(checkpoint['args'])
            path = get_dset_path(_args.dataset_name, self.dset_type)
            
            _args.pred_len = 12
            _args.batch_size = 1
            _args.dset = 'custom'
            path = '../dataset/test_sgan'
            
            timelength = len(close_humans[list(close_humans.keys())[0]])
            file_string = ''
            for t in range(timelength):
                if(t>self.obs_len):
                    continue
                for i in close_humans.keys():
                    file_string += str(t*10)+'\t'+str(i)+'.\t'+str(float(close_humans[i][(t)][0]))+'\t'+str(float(close_humans[i][(t)][1]))+'\n'

            for t in range(_args.pred_len):
                for i in close_humans.keys():
                    file_string += str(10*self.obs_len + t*10)+'\t'+str(i)+'.\t0.\t0.\n'


            with open('../dataset/test_sgan/SGan_past_trajectory.txt', 'w') as f:
                f.write(file_string)

            _, loader = data_loader(_args, path)
            
            time_start = time.time()
            prediction = self.evaluate(_args, loader, generator, self.num_samples)
            time_end = time.time()
            if(DEBUG):
                print(f'pure prediction time is {time_end-time_start} s.')    


        # Fills an output dictionary with the data from the prediction
        if(DEBUG):
            print(f'Prediction Horizon: {self.prediction_horizon}, Number of humans {len(close_humans)}')
        total_predictions=[]
        
        for key in prediction:
            # sinlge mode prediction
            dims = list(prediction[key].size())
            # first element is time steps
            time_predictions = dims[0]                
            # 

            # second element the number of agents
            num_agents = dims[1]
            # third element should be the number of coordinates
            num_coordinates = dims[2]
            predictions = np.zeros((time_predictions, num_agents, num_coordinates))
            for t in range(0, time_predictions):
                for n_agent in range(0, num_agents):
                    predictions[t, n_agent, 0] = prediction[key][t, n_agent, 0]
                    predictions[t, n_agent, 1] = prediction[key][t, n_agent, 1]
            total_predictions.append(predictions)

        
        return total_predictions, float(time_end-time_start)
        
    def predict(self, scenario):
        # This is the container for the predicted sample trajectories
        prediction = []
        self.prediction_horizon = scenario.prediction_horizon
        # Access the observed trajectories in this scenario
        # Adding just the last time frame per each human
        close_humans = {}
        for i in range(len(scenario.trajectories)):  # trajectory of person i
            agent_i_traj = scenario.trajectories[i][:, :]
            if(DEBUG):
                print(f'agent_i_traj {agent_i_traj}')
            close_humans[i] = []
            # including the last two time instants for prediction (needed for computing velocities)
            for agent_position in agent_i_traj:
                close_humans[i].append([agent_position[0], agent_position[1], 0.0, 0.0])
            # close_humans[i].append([agent_i_traj[-1, 0], agent_i_traj[-1, 1], 0.0, 0.0])
        
        if(DEBUG):
            print(f'Dict of close humans: {close_humans}')
            
        prediction, prediction_time = self.predictSGAN(close_humans)

        # Returning only the predictions for the horizon of the scenario
        prediction = [ k[:self.prediction_horizon, :, :]  for k in prediction]

        p = Prediction(prediction)
        p.runtime = prediction_time
        return p
