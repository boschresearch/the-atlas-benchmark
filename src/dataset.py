import sys

from pedestrian import Pedestrian
from data_loaders import read_map_from_json, trans_map, read_json, read_point_obstacles
import numpy as np
import copy
import matplotlib.pyplot as plt
from scenario import Scenario

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Dataset(object):

    def __init__(self, input_list, split=None):
        self.name = input_list['dataset']['name']
        self.downsampling = input_list['benchmark']['setup']['downsample']
        self.ds_rate = input_list['benchmark']['setup']['downsample rate']
        self.mapds = input_list['benchmark']['setup']['downsample map']
        self.mapds_rate = input_list['benchmark']['setup']['downsample map rate']

        self.added_noise_sigma = input_list['benchmark']['setup']['added_noise_sigma']
        self.interpolation = input_list['benchmark']['setup']['interpolate']
        self.smoothing = input_list['benchmark']['setup']['smooth']

        self.path = input_list['dataset']['data']
        self.annotation_interval = input_list['dataset']['annotation interval']
        self.frame_frequency = input_list['dataset']['frequency']
        self.map = self.import_map(input_list['dataset']['map'])
        self.goals = self.import_goals(input_list['dataset']['goals'])

        # Importing and concatenating multiple dataset files
        detection_arrays = []
        frame_counter = 0
        for file in self.path:
            if file[0] != '/':
                path = '/' + file
            else:
                path = file
            detection_array = read_json(ROOT_DIR + path)
            detection_array[:,0] += frame_counter
            frame_counter = 1.01*int(max(detection_array[:, 0]))
            detection_arrays.append(detection_array)
        self.detections = np.concatenate(tuple(detection_arrays))

        if split:
            l = len(self.detections)
            split_start = int(l*split[0])
            split_end = int(l*split[1])
            self.detections = self.detections[split_start:split_end,:]

        self.min_frame = int(min(self.detections[:, 0]))
        self.max_frame = int(max(self.detections[:, 0]))

        if self.downsampling:
            self.down_sample(self.ds_rate)

        if self.mapds:  # if map_ds, map should not be empty and should be real world map
            if self.obstacle_type == 1 and self.map != []:
                self.map = self.map[:, ::self.mapds_rate]

        self.pedestrians = self.extract_pedestrian_trajectories()

        if self.interpolation:
            self.interpolate()

        if self.smoothing != 0:
            self.smooth(self.smoothing)

        #if self.added_noise_sigma != 0:
        #    self.add_noise(self.added_noise_sigma)

    def import_goals(self, goal_path):
        if goal_path:
            return np.loadtxt(ROOT_DIR+goal_path)
        else:
            return []

    def import_map(self, maps):
        if maps:
            self.obstacle_type = maps['obstacle_type']
            if maps['type'] == 'pic':
                if 'H_matrix' in maps:
                    world_d2 = trans_map(ROOT_DIR+maps['picture'], ROOT_DIR+maps['H_matrix'])
                else:
                    world_d2 = trans_map(ROOT_DIR+maps['picture'], None)
            if maps['type'] == 'sem':
                world_d2, _ = read_map_from_json(ROOT_DIR+maps['picture'], ROOT_DIR+maps['sem_class'])  # path is to tell exp 1/2 or exp 3
            if maps['type'] == 'polygonal' and self.obstacle_type == 2:  # load center-radius map
                world_d2 = read_point_obstacles(ROOT_DIR+maps['picture'])
            return world_d2
        else:
            self.obstacle_type = 0
            return []

    def down_sample(self, downsample_rate):
        self.frame_frequency = self.frame_frequency / downsample_rate
        self.annotation_interval = self.annotation_interval * downsample_rate
        detections_ds = []
        frames = self.detections[:, 0].astype(int)
        frame_chosen = np.arange(min(frames), max(frames)+1, self.annotation_interval)
        # this algorithm may be improved
        for i in range(len(frames)):
            if frames[i] in frame_chosen:
                detections_ds.append(self.detections[i])
        self.detections = np.array(detections_ds)

    def add_noise(self, pedestrians_trajectories):
        #pedestrians_trajectories = copy.deepcopy([o.trajectory for o in self.pedestrians])
        result = []
        sigma = self.added_noise_sigma / self.frame_frequency
        for i in range(len(pedestrians_trajectories)):
            len_traj = len(pedestrians_trajectories[i])
            noise = np.random.normal(0, sigma, (len_traj, 2))
            result.append(pedestrians_trajectories[i] + noise)
        return result

    def interpolate(self):
        pedestrians_frames = [o.frames for o in self.pedestrians]
        pedestrians_trajectories = [o.trajectory for o in self.pedestrians]
        for i in range(len(pedestrians_frames)):
            if len(pedestrians_frames[i]) > 2:
                frame_traj = pedestrians_frames[i]
                frame1 = frame_traj[0:-2]
                frame2 = frame_traj[1:-1]
                frame_interval = frame2 - frame1
                min_interval = min(frame_interval)
                max_interval = max(frame_interval)
                if max_interval != min_interval:
                    # print('exist frame loss', 'in ped', ped_id[i])
                    frame_loss_time = 4  # which means, frame skip in 4 seconds, are considered as frame loss
                    exception_ind = np.where(frame_interval != min_interval)[0]
                    exception_frame = frame_interval[exception_ind]
                    traj_x = pedestrians_trajectories[i][:, 0]
                    traj_y = pedestrians_trajectories[i][:, 1]
                    frame_loss_count = 0
                    for j in range(len(exception_ind)):
                        max_frame_interval = np.floor(frame_loss_time * self.frame_frequency * self.annotation_interval)
                        if exception_frame[j] <= max_frame_interval:
                            ind = exception_ind[j]
                            frame_skip = exception_frame[j]
                            traj_x_new = np.linspace(traj_x[ind], traj_x[ind + 1],
                                                     int(frame_skip / self.annotation_interval + 1))
                            traj_y_new = np.linspace(traj_y[ind], traj_y[ind + 1],
                                                     int(frame_skip / self.annotation_interval + 1))
                            frame_new = np.linspace(frame_traj[ind], frame_traj[ind + 1],
                                                    int(frame_skip / self.annotation_interval + 1))
                            # self.Ped_detections.frame_ped[i] = np.insert(frame_traj, ind + 1, frame_new[1:-1])
                            pos_traj = np.array([traj_x_new[1:-1], traj_y_new[1:-1]]).T
                            # self.Ped_detections.frame_ped[i] =
                            pedestrians_frames[i] = np.insert(pedestrians_frames[i], ind + 1 + frame_loss_count, frame_new[1:-1])
                            pedestrians_trajectories[i] = np.insert(pedestrians_trajectories[i], ind + 1 + frame_loss_count, pos_traj,
                                                          axis=0)
                            frame_loss_count += len(frame_new) - 2

                            self.pedestrians[i].trajectory = pedestrians_trajectories[i]
                            self.pedestrians[i].frames = pedestrians_frames[i]

    # I think this function with average_length appeared after detecting the bug with smoothing in THOR
    # when the same pedestrian was disappearing and appearing in a different location
    def smooth(self, average_len):
        pedestrians_trajectories = [o.trajectory for o in self.pedestrians]
        pedestrians_frames = [o.frames for o in self.pedestrians]
        for i in range(len(pedestrians_trajectories)):
            if len(pedestrians_trajectories[i]) > 2:
                frame_traj = pedestrians_frames[i]
                frame1 = frame_traj[0:-2]
                frame2 = frame_traj[1:-1]
                frame_interval = frame2 - frame1
                min_interval = min(frame_interval)
                max_interval = max(frame_interval)
                if max_interval != min_interval:  # judge if the ped's trajectory is continuous
                    exception_ind = np.where(frame_interval != min_interval)[0]
                    list_ind = list(exception_ind)
                    list_ind.insert(0, -1)
                    for n in range(len(list_ind) - 1):
                        if (list_ind[n + 1] - list_ind[n]) > average_len:  # because use convolution
                            self.pedestrians[i].trajectory[list_ind[n] + 1:list_ind[n + 1] + 1] = \
                                self.smooth_trajectory(pedestrians_trajectories[i][list_ind[n] + 1:list_ind[n + 1] + 1], average_len)
                else:
                    if len(pedestrians_trajectories[i]) > average_len:  # because use convolution
                        self.pedestrians[i].trajectory = self.smooth_trajectory(pedestrians_trajectories[i], average_len)

    def smooth_trajectory(self, trajectory, points):
        box = np.ones(points) / points
        # use valid, np.floor(points/2) less points
        # use same, deal with edge error points
        loss_points = int(np.floor(points / 2))
        x = np.convolve(trajectory[:, 0], box, mode='same')
        x[0:loss_points] = trajectory[0:loss_points, 0]
        x[-loss_points::] = trajectory[-loss_points::, 0]
        y = np.convolve(trajectory[:, 1], box, mode='same')
        y[0:loss_points] = trajectory[0:loss_points, 1]
        y[-loss_points::] = trajectory[-loss_points::, 1]
        tra_smooth = np.vstack((x, y)).T
        return tra_smooth
    
    def extract_pedestrian_trajectories(self):
        pid = self.detections[:, 1] # IDs of the pedestrians
        list_ped = [] # Unique IDs of the pedestrians
        for i in pid:
            if not i in list_ped:
                list_ped.append(int(i))

        segmented_pedestrians = []
        for i in list_ped:
            p = self.detections[self.detections[:, 1] == i]
            if len(p) != 0:  # for some pedestrianID, there are no data stored [not same as original id]
                ped = Pedestrian()
                start_time = p[0][0]
                end_time = p[-1][0]
                pt = p[:, [2, 3]]
                frame_single = p[:, 0]
                ped.trajectory = pt
                ped.start_frame = start_time
                ped.end_frame = end_time
                ped.frames = frame_single
                ped.ped_id = i
                segmented_pedestrians.append(ped)
        return segmented_pedestrians

    # find indexes of frames in which the pedestrian is detected in the given interval [startframe, endframe]
    def detections_between_frames(self, startframe, endframe):
        pedestrians_frames = [o.frames for o in self.pedestrians] # vector of vectors
        pedestrians_frames_interval = [] # vector of vectors
        for i in range(len(pedestrians_frames)): # for each pedestrian i
            if len(pedestrians_frames[i]) != 0: # if this pedestrians is detected anywhere
                a = np.where((pedestrians_frames[i] >= startframe) & (pedestrians_frames[i] < endframe))
            else:
                a = np.where(0 > 1)
            pedestrians_frames_interval.append(a)
        return pedestrians_frames_interval

    # this function generates the testscene (observations and GT) at the given frame
    def generate_scenario(self, startframe, prediction_length, observation_len, complete_gt_required=True):
        pedestrians_frames = [o.frames for o in self.pedestrians] # vector of vectors
        pedestrians_trajectories = [o.trajectory for o in self.pedestrians] # vector of vectors
        pedestrians_ids = [o.ped_id for o in self.pedestrians] # vector of ints

        endframe = startframe + observation_len * self.annotation_interval
        pedestrians_frames_interval = self.detections_between_frames(startframe,endframe)

        scenario_trajectories = []  # trajectories of pedestrians with sufficient observation length
        scenario_pedestrian_indeces = [] # indexes of pedestrians with sufficient observation length
        scenario_pedestrian_ids = [] # IDs of pedestrians with with sufficient observation length
        # before, empty trajectory ped still contained; after, not contained
        for i in range(len(pedestrians_trajectories)): # for each pedestrian i
            # positions of pedestrian [i] between startframe and endframe
            position = pedestrians_trajectories[i][pedestrians_frames_interval[i]]
            # ensure each trajectory has enough observed positions
            if (len(position) >= observation_len):
                scenario_pedestrian_indeces.append(i)
                scenario_pedestrian_ids.append(pedestrians_ids[i])
                scenario_trajectories.append(position)
            elif (len(position) > 0):
                return [],[],[] # condition for determining invalid scenes with incomplete observations

        # now looking for available ground truth in the interval
        # [endframe, finalframe]
        finalframe = endframe + prediction_length * self.annotation_interval
        scenario_gt = []
        for i in scenario_pedestrian_indeces:
            if len(pedestrians_frames[i]) != 0:
                frame_gt_ind = self.detections_between_frames(endframe,finalframe)
                trajectory_GT = pedestrians_trajectories[i][frame_gt_ind[i]]
                if (len(trajectory_GT)<prediction_length) & complete_gt_required:
                    return [],[],[] # condition for determining invalid scenes with incomplete GT
                scenario_gt.append(trajectory_GT)
        return scenario_trajectories, scenario_pedestrian_ids, scenario_gt
    
    # Iterate over all possible start frames in the dataset and extract the valid scenarios
    # with sufficient observations for all detected pedestrians and the ground truth data
    def extract_scenarios(self, prediction_horizon, observation_len, min_num_prople=1):
        total_scenes = 0
        valid_scenes = []
        for startframe in range(self.min_frame, self.max_frame, self.annotation_interval):
            total_scenes = total_scenes + 1
            scenario_trajectories, scenario_pedestrian_ids, scenario_gt = \
                        self.generate_scenario(startframe, prediction_horizon, observation_len)
            scenario = Scenario(startframe, scenario_trajectories, scenario_pedestrian_ids, \
                        scenario_gt, prediction_horizon, observation_len)
            if(len(scenario.trajectories)>=min_num_prople):
                if(self.added_noise_sigma > 0):
                    scenario.trajectories = self.add_noise(scenario.trajectories)
                valid_scenes.append(scenario)

        print("The", self.name, "dataset has", len(valid_scenes), "valid scenes with observation length", observation_len, 
                "and prediction horizon", prediction_horizon, "out of", total_scenes, "scenes total.")
        return valid_scenes

    def plot_frame_skip(self):
        pedestrians_frames = [o.frames for o in self.pedestrians]
        pedestrians_ids = [o.ped_id for o in self.pedestrians]
        for i in range(0, len(pedestrians_frames)):
            plt.scatter(pedestrians_frames[i], i * np.ones(len(pedestrians_frames[i])),
                        label='%sth pedestrian' % (pedestrians_ids[i]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        return

    def plot_map(self):
        if self.obstacle_type==1:
            plt.scatter(self.map[0, :], self.map[1, :],s=0.2,color='lightgrey')
        elif self.obstacle_type==2:
            plt.scatter([o[0][0] for o in self.map],[o[0][1] for o in self.map], s=[1400*o[1] for o in self.map], color='lightgrey')
        return

    def plot_frame_period_old(self, startframe, duration):
        annotation_interval = self.annotation_interval
        pedestrians = self.pedestrians
        endframe = startframe + duration * annotation_interval
        self.plot_map()
        frame_tra, ped_frame, tra_GT = self.generate_scenario(startframe, 0, duration)
        start_time_frame = dict()
        for p in pedestrians:
            start_time_frame[p.ped_id] = max(p.start_frame,startframe)
        # for i in chosen_ped:
        for i in range(len(frame_tra)):
            plt.plot(frame_tra[i][0, 0], frame_tra[i][0, 1], 'o', color='olivedrab', label='start frame')
            plt.plot(frame_tra[i][:, 0], frame_tra[i][:, 1], '-*',
                    linewidth=1.5, label='%sth ped. (id %s)' % (i,ped_frame[i]))
            plt.annotate('%s' % int(start_time_frame[ped_frame[i]]), xy=(frame_tra[i][0, 0], frame_tra[i][0, 1]),
                        xytext=(-10, 5), textcoords='offset points')

        handles, labels = plt.gca().get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del (labels[i])
                del (handles[i])
            else:
                i += 1
        plt.title('Detections in frames %s to %s' % (startframe, endframe))
        plt.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.axis('equal')
        plt.show()
        return
    
    def plot_frame_period(self, startframe, duration):
        pedestrians_trajectories = [o.trajectory for o in self.pedestrians] # vector of vectors
        pedestrians_ids = [o.ped_id for o in self.pedestrians] # vector of ints
        annotation_interval = self.annotation_interval
        endframe = startframe + duration * annotation_interval
        self.plot_map()
        start_time_frame = dict()
        for p in self.pedestrians:
            start_time_frame[p.ped_id] = max(p.start_frame,startframe)
        pedestrians_frames_interval = self.detections_between_frames(startframe,endframe)
        scenario_trajectories = []  # trajectories of pedestrians with sufficient observation length
        scenario_pedestrian_indeces = [] # indexes of pedestrians with sufficient observation length
        scenario_pedestrian_ids = [] # IDs of pedestrians with with sufficient observation length
        # before, empty trajectory ped still contained; after, not contained
        for i in range(len(pedestrians_trajectories)): # for each pedestrian i
            # positions of pedestrian [i] between startframe and endframe
            position = pedestrians_trajectories[i][pedestrians_frames_interval[i]]
            # ensure each trajectory has enough observed positions
            scenario_pedestrian_indeces.append(i)
            scenario_pedestrian_ids.append(pedestrians_ids[i])
            scenario_trajectories.append(position)
        # for i in chosen_ped:
        for i in range(len(scenario_trajectories)):
            if(scenario_trajectories[i].shape[0]>0):
                plt.plot(scenario_trajectories[i][0, 0], scenario_trajectories[i][0, 1], 'o', color='olivedrab', label='start frame')
                plt.plot(scenario_trajectories[i][:, 0], scenario_trajectories[i][:, 1], '-*',
                        linewidth=1.5, label='%sth ped. (id %s)' % (i,scenario_pedestrian_ids[i]))
                plt.annotate('%s' % int(start_time_frame[scenario_pedestrian_ids[i]]), xy=(scenario_trajectories[i][0, 0], scenario_trajectories[i][0, 1]),
                            xytext=(-10, 5), textcoords='offset points')

        handles, labels = plt.gca().get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del (labels[i])
                del (handles[i])
            else:
                i += 1
        plt.title('Detections in frames %s to %s' % (startframe, endframe))
        plt.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.axis('equal')
        plt.show()
        return
    
    def plot_ped(self, n, m):
        self.plot_map()
        pedestrians_trajectories = [o.trajectory for o in self.pedestrians]
        pedestrians_sfs = [o.start_frame for o in self.pedestrians]
        pedestrians_efs = [o.end_frame for o in self.pedestrians]
        pedestrians_ids = [o.ped_id for o in self.pedestrians]
        start_frame = self.pedestrians[n].frames[0]
        end_frame = self.pedestrians[m-1].frames[-1]
        print('%s - %s' % (int(start_frame), int(end_frame)))
        for i in range(n, m):
            if len(pedestrians_trajectories[i]) > 0:
                plt.plot(pedestrians_trajectories[i][0, 0], pedestrians_trajectories[i][0, 1], 'o', color='palevioletred', label='start point')
                plt.annotate('%s' % int(pedestrians_sfs[i]), xy=(pedestrians_trajectories[i][0, 0], pedestrians_trajectories[i][0, 1]),
                            xytext=(-10, 5), textcoords='offset points')
                plt.plot(pedestrians_trajectories[i][-1, 0], pedestrians_trajectories[i][-1, 1], 'o', color='olivedrab', label='end point')
                plt.annotate('%s' % int(pedestrians_efs[i]), xy=(pedestrians_trajectories[i][-1, 0], pedestrians_trajectories[i][-1, 1]),
                            xytext=(-10, 5), textcoords='offset points')
                plt.plot(pedestrians_trajectories[i][:, 0], pedestrians_trajectories[i][:, 1], '*-', label='%sth ped. (id %s)' % (i, pedestrians_ids[i]))
        handles, labels = plt.gca().get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del (labels[i])
                del (handles[i])
            else:
                i += 1
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if(m-n==1):
            plt.title('trajectory of pedestrian %s' % int(n))
        else:
            plt.title('trajectory of pedestrians %s to %s' % (int(n), int(m-1)))
        plt.axis('equal')
        plt.show()
        return