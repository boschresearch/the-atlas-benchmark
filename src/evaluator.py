# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0



from matplotlib.markers import MarkerStyle
import numpy as np
from numpy import full
from sklearn import mixture
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from matplotlib import animation, axes
import random
import colorsys
from matplotlib.patches import Ellipse

def get_mean_std(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return mean, std


class Evaluator:
    def __init__(self, set_GT_length):
        self.set_GT_length = set_GT_length

    def draw_ellipse(self, position, covariance, color='r', ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ellipse = Ellipse(position, nsig * width, nsig * height,
                                angle, color=color, **kwargs)
            ax.add_patch(ellipse)

    def plot_gt(self,ax,scenario, people_colors):
        num_people = np.size(scenario.gt, axis=0)
        for i in range(num_people):  # i is number of ped
            ax.plot(scenario.trajectories[i][:, 0], scenario.trajectories[i][:, 1], '.',linewidth=1,markersize=1,color=people_colors[i][0],label='Observations')
            ax.plot(scenario.gt[i][:, 0], scenario.gt[i][:, 1], 'D', linewidth=1,markersize=1,color=people_colors[i][1],label='Ground truth')

    def plot_predictions(self,ax,full_predictions, people_colors):
        num_people = full_predictions.trajectories[0].shape[1]
        num_pred_steps = full_predictions.trajectories[0].shape[0]
        for prediction_sample in full_predictions.trajectories:
                for i in range(num_people):  # i is number of ped
                    pred_traj = prediction_sample[:num_pred_steps, i, :]
                    ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'o',linewidth=1,markersize=2,color=people_colors[i][2],label='Predictions')
    
    # Prepares an array of distinct random colors for the N people of this scenario
    # Output structure, one row per person:
    # [ [color_obs,color_gt,color_pred],
    #   [color_obs,color_gt,color_pred],
    #   ...
    #   [color_obs,color_gt,color_pred] ]
    def prepare_scenario_colorscheme(self, scenario):
        num_people = np.size(scenario.gt, axis=0)

        HSV_tuples_obs = [(x*1.0/num_people, 0.5, 0.4) for x in range(num_people)]
        RGB_tuples_obs = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples_obs))
        HSV_tuples_gt = [(x*1.0/num_people, 0.97, 0.7) for x in range(num_people)]
        RGB_tuples_gt = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples_gt))
        HSV_tuples_pred = [(x*1.0/num_people, 0.5, 0.95) for x in range(num_people)]
        RGB_tuples_pred = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples_pred))

        colors = [[c1,c2,c3] for c1,c2,c3 in zip(RGB_tuples_obs,RGB_tuples_gt,RGB_tuples_pred)]
        random.shuffle(colors)
        return colors

    def plot_scenario(self, dataset, scenario, full_predictions=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        people_colors = self.prepare_scenario_colorscheme(scenario)
        
        dataset.plot_map()
        if full_predictions:
            self.plot_predictions(ax,full_predictions, people_colors)
        self.plot_gt(ax, scenario, people_colors)

        if full_predictions and len(full_predictions.trajectories) > 1:
            _, gauss_scene = self.cal_nll(scenario,full_predictions)
            gauss_scene = np.array(gauss_scene)
            for pos, covar, color in zip(gauss_scene[:, 1], gauss_scene[:, 0],people_colors):
                self.draw_ellipse(pos, covar, color[2], alpha=0.3)
        
        handles, labels = self.filter_legend(ax)
        plt.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.axis('equal')
        plt.show()
        return

    def filter_legend(self,ax):
        handles, labels = ax.get_legend_handles_labels()
        i = 1
        while i < len(labels):
            if labels[i] in labels[:i]:
                del (labels[i])
                del (handles[i])
            else:
                i += 1
        return handles, labels

    def draw_animation(self,dataset,scenario,prediction):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        num_samples = len(prediction.trajectories)
        num_people = prediction.trajectories[0].shape[1]
        people_colors = self.prepare_scenario_colorscheme(scenario)

        dataset.plot_map()
        self.plot_gt(ax,scenario,people_colors)
        
        dots = []
        for i in range(num_people):
            for s in range(num_samples):
                #ax.plot(prediction[s][:,i,0], prediction[s][:,i,1], color=people_colors[i][2]) # Plot the trajectories of each person
                dot, = ax.plot([], [], '.',linewidth=1,markersize=5, color=people_colors[i][2], label='Predictions')
                dots.append(dot) # Prepare the dots to be animated for each person

        handles, labels = self.filter_legend(ax)
        ax.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        ax.axis('equal')
        #plt.title('helbing')

        def init():
            start = []
            for i in range(num_people):
                for s in range(num_samples):
                    s = ax.plot(prediction.trajectories[s][0,i,0], prediction.trajectories[s][0,i,1], color=people_colors[i][2])
                    start.append(s)
            return start

        def update_dot(j):
            counter = 0
            for i in range(num_people):
                for s in range(num_samples):
                    dots[counter].set_data(prediction.trajectories[s][j,i,0], prediction.trajectories[s][j,i,1])
                    counter = counter+1
            return dots

        ani = animation.FuncAnimation(fig, update_dot, frames=range(0, np.size(prediction.trajectories[0], axis=0)), interval=500, init_func=init)
        return ani

    # This function calculates ADE for single trajectory
    def cal_ade(self, predict, true):  # traj should be (pre_step,2)
        squared_dist = (predict - true) ** 2
        temp = squared_dist.sum(axis=1)
        ade = np.sqrt(temp).sum()
        ade = ade / len(predict)
        return ade

    # This function calculates FDE for single trajectory
    def cal_fde(self, predict, true):
        squared_dist = (predict - true) ** 2
        fde = np.sqrt(squared_dist[-1, :].sum())
        return fde

    # This function calculates the ADE and FDE for:
    # - certain predictions with one particle
    # - uncertain predictions with multiple samples
    def evaluate_scenario_ade_fde(self, scenario, predictions):
        ade = []
        fde = []
        for prediction in predictions.trajectories:
            ade_single, fde_single = self.evaluate_single_particle_ade_fde(scenario, prediction)
            ade.append(ade_single)
            fde.append(fde_single)
        if len(ade) != 0:
            ade = np.mean(ade)
            fde = np.mean(fde)
        else:
            ade = np.nan
            fde = np.nan
        return ade, fde

    # Evaluate the predictions for a scenario
    def evaluate_single_particle_ade_fde(self, scenario, predictions):
        ade = []
        fde = []
        for i in range(np.size(scenario.gt, axis=0)):  # i is number of ped
            if len(scenario.gt[i]) >= self.set_GT_length:
                true_length = len(scenario.gt[i])
                ade_single = self.cal_ade(predictions[:true_length, i, :], scenario.gt[i])
                fde_single = self.cal_fde(predictions[:true_length, i, :], scenario.gt[i])
                ade.append(ade_single)
                fde.append(fde_single)
        if len(ade) != 0:
            ade = np.mean(ade)
            fde = np.mean(fde)
        else:
            ade = np.nan
            fde = np.nan
        # if ade=[], return nan; else, return float
        return ade, fde

    # Calculating the kADE and kFDE for a single pedestrian
    # tra_gt is a single ground truth trajectory
    # predict contains multiple noisy trajectories
    def k_ade_fde_pointwise(self, predict, tra_gt):
        k_ade = []
        k_fde = -1
        len_gt = len(tra_gt)
        if len_gt >= self.set_GT_length:
            for i in range(len_gt):
                timestep_pred = predict[:,i,:]
                min_dist = min(np.sqrt(((timestep_pred-tra_gt[i,:])**2).sum(axis=1)))
                k_ade.append(min_dist)
                k_fde = min_dist
        return np.mean(k_ade), k_fde

    # Evaluate the predictions for a scenario
    def evaluate_scenario_kade_kfde(self, scenario, predictions):
        kADE = []
        kFDE = []
        for i in range(len(scenario.gt)): # for each pedestrian i
            single_ped_prediction =  np.array([sample[:,i,:] for sample in predictions.trajectories])
            kade, kfde = self.k_ade_fde_pointwise(single_ped_prediction, scenario.gt[i])
            if not isinstance(kade, list):
                kADE.append(kade)
                kFDE.append(kfde)
        return np.mean(kADE), np.mean(kFDE)

    # Input is a prediction for a single pedestian
    # Model is fit to the last points
    def calculate_gmm(self, samples, gt_position):
        model = mixture.GaussianMixture()
        fitted_values = model.fit(samples)
        predicted_values = model.predict(samples)
        # compute centers as point of highest density of distribution
        density = mvn(cov=model.covariances_[0], mean=model.means_[0]).logpdf(samples)
        centers = samples[np.argmax(density)]

        nll = - mvn(cov=model.covariances_[0], mean=model.means_[0]).logpdf(gt_position)
        return model.covariances_[0], model.means_[0], centers, nll

    # input is un_sof(scene)
    def cal_nll(self, scenario, prediction):
        if len(prediction.trajectories) < 2:
            print("NLL calculation is not possible for a prediction with one sample.")
            return -1
        num_ped = prediction.trajectories[0].shape[1]
        gauss_scene = []
        nll_scene = []
        for ped_id in range(num_ped):
            if (len(scenario.gt[ped_id]) >= self.set_GT_length) :
                # Selecting of the single person trajectory, [sample,time,position]
                chosen_ped = np.array([sample[:,ped_id,:] for sample in prediction.trajectories])
                len_gt = len(scenario.gt[ped_id])
                gt_position = scenario.gt[ped_id][-1]
                # Selecting of the single person trajectory at the last gt point in time, [sample,position]
                samples = chosen_ped[:, len_gt - 1, :]
                sigma, mu, centers, nll = self.calculate_gmm(samples, gt_position)
                gauss_scene.append([sigma, mu, centers, nll])
                nll_scene.append(nll)
        nll_scene_mean = np.mean(nll_scene)
        return nll_scene_mean, gauss_scene
