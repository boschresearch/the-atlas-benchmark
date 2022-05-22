
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Trajectron-plus-plus/trajectron')))

print(sys.path)
# Imports from Trajectron++
from model.online.online_trajectron import OnlineTrajectron
from model.model_registrar import ModelRegistrar
from environment import Environment, Scene, Node, derivative_of


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEBUG = False

class TrajectronPredictor(Predictor):
    '''
    This class is the predictor that outputs the predicted future human trajectories that are used in the controller
    Input:

    the model parameters determine what model is used for the predictions. 
    see the folder 'trajectronplusplus/models' for options.

    close_humans:
    A dictionary with all the observed N humans in the area. The keys will be numbers and the values 
    will be np.arrays with the state with [x, y, vx, vy].

    states_list:
    A list to hold all the past robot states [x, y, vx, vy]
    '''

    def __init__(self,
                 scenario,
                 dataset,
                 method_cfg={},
                 num_modes=25,
                 model_full_path='../../Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3',
                 model_checkpoint=100,
                 model_nodetype='Pedestrian'
                 ):
        Predictor.__init__(self, dataset)
        self.dt = self.delta_T 
        self.num_modes = num_modes  # trajectron only outputs 25 modes
        self.prediction_horizon = scenario.prediction_horizon
        self.model_full_path = model_full_path
        self.model_checkpoint = model_checkpoint
        self.model_nodetype = model_nodetype
        self.cnt_time_steps = 0
        self.max_scene_list_size = 400
        self.mean_x = 0
        self.mean_y = 0
        self.scenario = scenario
        if(method_cfg):
            self.uncertainty = method_cfg['uncertainty']
            self.uncertainty_sigma = method_cfg['uncertainty sigma']
            self.num_particles = method_cfg['num_particles']
        else:
            self.uncertainty = False
            self.uncertainty_sigma = 0.1
            self.num_particles = 1


        # This is needed to standardize the pedestrian locations
        self.standardization = {
            'PEDESTRIAN': {
                'position': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1}
                },
                'velocity': {
                    'x': {'mean': 0, 'std': 2},
                    'y': {'mean': 0, 'std': 2}
                },
                'acceleration': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1}
                }
            }
        }
        
        
        model_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_full_path))
        if(DEBUG):
            print(model_abs_path)
        with open(os.path.join(model_abs_path, 'config.json'), 'r') as config_json:
            self.hyperparams = json.load(config_json)      
            
        # builds an initial trajectron environment
        # This is done by building an "empty" dummy environment,
        # and loading the trajectron predictor with its configuration.
        # Later, different data can be inserted
        # scene used to append human information
        self.scene = Scene(timesteps=self.hyperparams['maximum_history_length'],
                               dt=self.dt, name="scene_0")
        
        
        # Adding just the initial time frame
        close_humans = {}
        for i in range(len(scenario.trajectories)):  # trajectory of person i
            agent_i_traj = scenario.trajectories[i][:, :]
            close_humans[i] = []
            for position in agent_i_traj[:-1]:
                close_humans[i].append([position[0], position[1], 0.0, 0.0])
        
        
        self.init_trajectron_env(close_humans)
        
        init_timestep = 1
        self.scene.nodes = self.scene.get_nodes_clipped_at_time(
            timesteps=np.arange(init_timestep - self.hyperparams['maximum_history_length'],
                                init_timestep + 1),
            state=self.hyperparams['state'])
        self.scene.calculate_scene_graph(attention_radius=self.env.attention_radius,
                                           edge_addition_filter=self.hyperparams['edge_addition_filter'],
                                           edge_removal_filter=self.hyperparams['edge_removal_filter'])

        # Load Model and set environment
        model_registrar = ModelRegistrar(model_abs_path, 'cpu')
        model_registrar.load_models(iter_num=model_checkpoint)
        self.trajectron = OnlineTrajectron(
            model_registrar, self.hyperparams, 'cpu')
        self.trajectron.set_environment(self.env, init_timestep)

    def init_trajectron_env(self, close_humans):
        '''
        close_humans is a dict of humans that should be considered for prediction.
        See robot class.
        '''
        self.env = Environment(
            node_type_list=['PEDESTRIAN'], standardization=self.standardization)
        attention_radius = dict()
        attention_radius[(self.env.NodeType.PEDESTRIAN,
                          self.env.NodeType.PEDESTRIAN)] = 3.0
        self.env.attention_radius = attention_radius

        return self.addNodes(close_humans)

    def addNodes(self, close_humans):
        '''
        Adding new humans to the scene graph as new nodes. 
        We assume that we can further increase the nodes in the scene graph on the fly 
        '''
        # Converts the self.humans_state into a trajectron environment made up of nodes and represented by a graph
        data_columns = pd.MultiIndex.from_product(
            [['position', 'velocity', 'acceleration'], ['x', 'y']])
 
        # creating a list of human positions
        scenelist = []
        num_time_steps = 0
        for key, value in close_humans.items():
            for i, j in enumerate(value):
                # the frame should consider also of a global time counter
                if(DEBUG):
                    print(f'{i}-{j}')
                if(i>num_time_steps):
                    num_time_steps = i
                scenelist.append(
                    np.array([self.cnt_time_steps + i, key, j[0], j[1], j[2], j[3]]))

        # creating the panda data frame
        scenearray = np.array(scenelist)
        
        if(len(scenelist)<1):
            return self.scene
        
        data = pd.DataFrame(scenearray)
        data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'v_x', 'v_y']
        data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
        data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
               
        self.cnt_time_steps = self.cnt_time_steps + (num_time_steps+1)
        if(DEBUG):
            print(f'cnt_time_steps is {self.cnt_time_steps} and num_time_steps {num_time_steps}')
        
        # data['frame_id'] -= data['frame_id'].min()
        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['track_id'].astype(str)
        data.sort_values('frame_id', inplace=True)
        if(DEBUG):
            print(data)
        # Assumption to be centered around zero
        # as in https://github.com/StanfordASL/Trajectron-plus-plus/blob/ef0165a93ee5ba8cdc14f9b999b3e00070cd8588/experiments/pedestrians/process_data.py
        self.mean_x = data['pos_x'].mean()
        self.mean_y = data['pos_y'].mean()
        data['pos_x'] = data['pos_x'] - self.mean_x
        data['pos_y'] = data['pos_y'] - self.mean_y

        # Now we add the nodes to the existing scene
        # For each node in the scene, add the human data.
        for node_id in pd.unique(data['node_id']):
            node_df = data[data['node_id'] == node_id]
            node_values = node_df[['pos_x', 'pos_y', 'v_x', 'v_y']].values
            new_first_idx = node_df['frame_id'].iloc[0]
            if(DEBUG):
                print(f'new_first_idx: {new_first_idx}')
            x = node_values[:, 0]
            y = node_values[:, 1]            
            vx = derivative_of(x, self.delta_T)
            vy = derivative_of(y, self.delta_T)
            ax = derivative_of(vx, self.delta_T)
            ay = derivative_of(vy, self.delta_T)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns)
            node = Node(node_type=self.env.NodeType.PEDESTRIAN,
                        node_id=node_id, data=node_data)
            node.first_timestep = new_first_idx
            self.scene.nodes.append(node)

        N = len(close_humans)*100
        self.scene.nodes = self.scene.nodes[-N:] #keep last 100 elements
        self.env.scenes = [self.scene]

        return self.scene

    def predictTrajectron(self, close_humans):
        '''
        predicts the future movement of the humans given
        '''
        # Our scene should be used as a container of the last human poses
        # Adding new humans as new nodes of the scene graph
        self.addNodes(close_humans)
        if(DEBUG):
            print(f'Adding nodes at time step {self.cnt_time_steps}')
            print(self.scene.nodes)
        # creating the new scene graph based on the new data
        self.scene.calculate_scene_graph(attention_radius=self.env.attention_radius,
          edge_addition_filter=self.hyperparams['edge_addition_filter'],
          edge_removal_filter=self.hyperparams['edge_removal_filter'])
            
        # getting just the most recent data for incrementally predict via the LSTM    
        input_dict = self.scene.get_clipped_input_dict(
            timestep=self.cnt_time_steps-1, state=self.hyperparams['state'])
                
        self.env.scenes = [self.scene]
        self.trajectron.set_environment(self.env, self.cnt_time_steps-1)
        if(DEBUG):
            print(f'Preparing scene at time step {self.cnt_time_steps-1}')
        
        
        # Parameters for trajectron
        maps = None
        robot_present_and_future = None
        start = time.time()
        dists, full_predictions = self.trajectron.incremental_forward(input_dict,
                                                                   maps,
                                                                   prediction_horizon=self.prediction_horizon,
                                                                   num_samples=self.num_particles,
                                                                   robot_present_and_future=robot_present_and_future,
                                                                   full_dist=True)

        end = time.time()
        if(DEBUG):
            print(f'pure prediction time is {end-start} s.')       
        # incrementing the global counter of the time steps
        # self.cnt_time_steps = self.cnt_time_steps + 1

        # Fills an output dictionary with the data from the prediction
        if(DEBUG):
            print(f'Prediction Horizon: {self.prediction_horizon}, Number of humans {len(close_humans)}')
        total_predictions=[]
        
        for i in range(0, self.num_particles):
            predictions = np.zeros((self.prediction_horizon, len(close_humans), 2))
            for t in range(0, self.prediction_horizon):
                for key, value in full_predictions.items():
                    if(DEBUG):
                        print(f'Key: {key}')
                        print(f'Pedestrian: {key.id}')
                        print(f'Value: {value}')
                        print(f'Value {value[0,i,t]} at time {t}')
                    predictions[t, int(key.id), :] = value[0,i,t]
                    predictions[t, int(key.id), 0] = predictions[t, int(key.id), 0] + self.mean_x
                    predictions[t, int(key.id), 1] = predictions[t, int(key.id), 1] + self.mean_y
            
            total_predictions.append(predictions)
    

        # Alternatively one can do it manually
        # for node, pred_dist in dists.items():
        #     if pred_dist.mus.shape[:2] != (1, 1):
        #         return

        #     means = pred_dist.mus.squeeze().cpu().numpy()
        #     covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        #     pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()
        #     #print(f'pis: {pis}')
        #     for timestep in range(means.shape[0]):
        #         # taking the most likely mode
        #         z_val = np.argmax(pis[timestep])
        #         #for z_val in range(means.shape[1]):
        #         if(timestep==0):
        #             predictions[timestep, int(node.id), 0] = float(close_humans[int(node.id)][-1][0])
        #             predictions[timestep, int(node.id), 1] = float(close_humans[int(node.id)][-1][1])
                                
        #         if(timestep>0):
        #             # taking most likely distribution
        #             mean = means[timestep, z_val]
        #             covar = covs[timestep, z_val]
        #             print(mean)
        #             predictions[timestep, int(node.id), 0] = mean[0] + self.mean_x
        #             predictions[timestep, int(node.id), 1] = mean[1] + self.mean_y
        #             print(f'Distribution {mean}, {covar}, {z_val}')
    
        return total_predictions
        
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
            close_humans[i].append([agent_i_traj[-2, 0], agent_i_traj[-2, 1], 0.0, 0.0])
            close_humans[i].append([agent_i_traj[-1, 0], agent_i_traj[-1, 1], 0.0, 0.0])
        
        if(DEBUG):
            print(f'Dict of close humans: {close_humans}')
            
        prediction = self.predictTrajectron(close_humans)

        
        return Prediction(prediction)