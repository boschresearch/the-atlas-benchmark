import math
import numpy as np
from predict.predictor import Predictor_CVM
from predict.assign_goal import get_goal
import copy

import sys
sys.path.append('..')
from prediction import Prediction

def g(distance):
    return max(0,distance)

def limit_acceleration(velocity, v0):
    max_v = 1.3 * np.linalg.norm(v0)
    if np.linalg.norm(velocity) <= max_v:
        new_v = velocity
    else:
        new_v = max_v * velocity / np.linalg.norm(velocity)
    return new_v

def exists(obj, chain):
    _key = chain.pop(0)
    if _key in obj:
        return exists(obj[_key], chain) if chain else obj[_key]

class Predictor_sof(Predictor_CVM):
    def __init__(self, dataset, method_cfg, parameters=['default']):
        Predictor_CVM.__init__(self, dataset, method_cfg)
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'm']):
            self.m = method_cfg['param']['optimal'][parameters[1]]['m']
        else:
            self.m = method_cfg['param']['default']['m']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'tau']):
            self.tau = method_cfg['param']['optimal'][parameters[1]]['tau']
        else:
            self.tau = method_cfg['param']['default']['tau']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'r_i']):
            self.r_i = method_cfg['param']['optimal'][parameters[1]]['r_i']
        else:
            self.r_i = method_cfg['param']['default']['r_i']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'r_ij']):
            self.r_ij = method_cfg['param']['optimal'][parameters[1]]['r_ij']
        else:
            self.r_ij = method_cfg['param']['default']['r_ij']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'K']):
            self.K = method_cfg['param']['optimal'][parameters[1]]['K']
        else:
            self.K = method_cfg['param']['default']['K']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'Ko']):
            self.Ko = method_cfg['param']['optimal'][parameters[1]]['Ko']
        else:
            self.Ko = method_cfg['param']['default']['Ko']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'sf_anisotropic_lambda']):
            self.sf_anisotropic_lambda = method_cfg['param']['optimal'][parameters[1]]['sf_anisotropic_lambda']
        else:
            self.sf_anisotropic_lambda = method_cfg['param']['default']['sf_anisotropic_lambda']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'future_horizon']):
            self.future_horizon = method_cfg['param']['optimal'][parameters[1]]['future_horizon']
        else:
            self.future_horizon = method_cfg['param']['default']['future_horizon']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'a']):
            self.a = method_cfg['param']['optimal'][parameters[1]]['a']
        else:
            self.a = method_cfg['param']['default']['a']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'b']):
            self.b = method_cfg['param']['optimal'][parameters[1]]['b']
        else:
            self.b = method_cfg['param']['default']['b']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'ao']):
            self.ao = method_cfg['param']['optimal'][parameters[1]]['ao']
        else:
            self.ao = method_cfg['param']['default']['ao']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'bo']):
            self.bo = method_cfg['param']['optimal'][parameters[1]]['bo']
        else:
            self.bo = method_cfg['param']['default']['bo']

    def calculate_normal_fij(self, position_i, position_j, n_i):
        destination_vector = position_i - position_j  # vector from j point to i
        d_ij = np.linalg.norm(destination_vector)
        n_ij = destination_vector / (d_ij + 0.00000001)  # the direction of f-ij
        cos_fai_ij = np.dot(n_i, n_ij)
        w = self.sf_anisotropic_lambda + (1 - self.sf_anisotropic_lambda) * (1 + cos_fai_ij) / 2
        # f_ij = w * ((self.a * np.exp(-abs(self.r_ij - d_ij) / self.b)) + self.K * g(self.r_ij - d_ij)) * n_ij
        f_ij = w * (self.a * np.exp(-abs(self.r_ij - d_ij) / self.b)) * n_ij
        f_ij_phy = self.K * g(self.r_ij - d_ij) * n_ij
        return f_ij, f_ij_phy

    def calculate_fio(self, position_i, p_min, r_io, n_i):  # not contain w
        destination_vector = position_i - p_min  # vector from j point to i
        d_io = np.linalg.norm(destination_vector)
        n_io = destination_vector / d_io  # the direction of f-ij
        cos_fai_ij = np.dot(n_i, n_io)
        # f_io = (2 * self.a * np.exp(-abs(r_io - d_io) / (2 * self.b)) + self.K * g(r_io - d_io)) * n_io
        # f_io = (self.ao * np.exp(-abs(r_io - d_io) / self.bo) + self.Ko * g(r_io - d_io)) * n_io
        w = self.sf_anisotropic_lambda + (1 - self.sf_anisotropic_lambda) * (1 + cos_fai_ij) / 2
        f_io = w * (self.ao * np.exp(-abs(r_io - d_io) / self.bo)) * n_io
        f_io_phy = self.Ko * g(r_io - d_io) * n_io
        return f_io, f_io_phy

    def predict_sf(self, position, velocity, v_0):
        new_velocity = []
        new_position = []
        for i in range(len(position)):
            # v_0 = velocity[i]  # velocity[i]= [vx,vy]
            n_i = velocity[i] / (np.linalg.norm(velocity[i]) + 0.000001)
            f_i = self.m * 1 / self.tau * (v_0[i] - velocity[i])  # now v0 = v[i]
            f_ij_sum = []
            f_ij_phy_sum = []
            for j in range(len(position)):  # calculate f_ij
                if j != i:
                    f_ij, f_ij_phy = self.calculate_normal_fij(position[i], position[j], n_i)
                    f_ij_sum.append(f_ij)
                    f_ij_phy_sum.append(f_ij_phy)
            f_ij_total = np.sum(f_ij_sum, axis=0) + np.sum(f_ij_phy_sum, axis=0)
            if self.obstacle_type == 1:
                d = np.sqrt(np.sum((position[i] - self.world_d2.T) ** 2, axis=1))
                d_min_index = np.argmin(d)  # find the nearest point in obstacles
                p_min = self.world_d2[:, d_min_index]
                r_io = self.r_ij
                f_io, f_io_phy = self.calculate_fio(position[i], p_min, r_io, n_i)
            elif self.obstacle_type == 2:
                pos_obs = np.array([e[0] for e in self.world_d2]) # positions of the obstacles
                d = np.sqrt(np.sum((position[i] - pos_obs) ** 2, axis=1))
                d_min_index = np.argmin(d)
                near_ob = self.world_d2[d_min_index][0]
                r = self.world_d2[d_min_index][1] # radius of the first obstacle
                r_io = self.r_i + r
                # only consider the nearest obstacle
                f_io, f_io_phy = self.calculate_fio(position[i], near_ob, r_io, n_i)
                f_io = 1.5 * f_io
            else:
                f_io = 0  # calculate the repulsive force from obstacles
                f_io_phy = 0
            f = f_i + f_ij_total + f_io + f_io_phy
            new_velocity_i = velocity[i] + f / self.m * self.delta_T
            new_velocity_i = limit_acceleration(new_velocity_i, v_0[i])
            new_position_i = position[i] + new_velocity_i * self.delta_T
            new_velocity.append(new_velocity_i)
            new_position.append(new_position_i)
        return new_position, new_velocity

    def predict_single(self, scenario):
        frame_tra = scenario.trajectories
        position = []  # the position of last history timestep, if 2 trajectories, 2 lists, each list is ndarray (2,)
        velocity = []
        v0 = []
        goal_position = self.get_goal_short(frame_tra)
        goal_position_l = copy.deepcopy(goal_position)
        if self.load_goal_file:
            goal_in_frame = get_goal(frame_tra, self.goal_list)
            for i in range(len(goal_in_frame)):
                if goal_in_frame[i] != 10:
                    goal_position_l[i] = self.goal_list[goal_in_frame[i], :]
        for i in range(len(frame_tra)):  # predict for i-ped in this frame
            # if (len(frame_tra[i]) >= 2):  # choose trajectory of ped[i] in this frame(most is [])
            old_velocity = (frame_tra[i][-1, :] - frame_tra[i][-2, :]) / self.delta_T
            old_position = frame_tra[i][-1, :]
            mean_velocity = self.average_velocity(frame_tra[i])
            position.append(old_position)
            velocity.append(old_velocity)
            v0.append(mean_velocity)
        pre_position = []
        pre_velocity = []
        v0 = np.squeeze(v0, axis=1)  # v0 is intended velocity, ndarray(# of ped, 2)
        # x = np.size(v0,1)
        # v1 = np.expand_dims(v0,axis=0)
        # en1 = np.size(v1,axis=1)
        velocity = v0
        for _ in range(scenario.prediction_horizon):  # generate new position for each trajectory of ped
            new_position, new_velocity = self.predict_sf(position, velocity, v0)
            if self.uncertainty:
                noise = np.random.normal(0, self.noise_scale / self.frame_frequency, (len(new_position), 2))
                new_position = new_position + noise
            pre_position.append(new_position)
            pre_velocity.append(new_velocity)
            position, velocity = new_position, new_velocity
            if self.consider_goals:
                if self.load_goal_file:
                    position = np.array(position)
                    direction_intend_l = goal_position_l - position
                    direction_intend_d = np.linalg.norm(direction_intend_l, axis=-1)[:, np.newaxis]
                    n_intend = direction_intend_l / (direction_intend_d ** 2 + 0.00001)
                    # n_intend = direction_intend_l / (np.linalg.norm(direction_intend, axis=-1)**2 + 0.00001)
                    v0_direction = v0 / (np.linalg.norm(v0, axis=-1)[:, np.newaxis] + 0.00001)
                    v_final = v0_direction + n_intend
                    v_final_direction = v_final / (np.linalg.norm(v_final, axis=-1)[:, np.newaxis] + 0.00001)
                    v0 = v_final_direction * np.linalg.norm(v0, axis=-1)[:, np.newaxis]
                    if 0:
                        import matplotlib.pyplot as plt
                        # world_d2 = dataset.map
                        plt.scatter(self.world_d2[0, :], self.world_d2[1, :])
                        # plt.figure()
                        ax = plt.gca()
                        X = position[:, 0]
                        Y = position[:, 1]
                        # U = goal_position_l[:,0]
                        # V = goal_position_l[:,1]
                        U = n_intend[:, 0]
                        V = n_intend[:, 1]
                        v_x = v0_direction[:, 0]
                        v_y = v0_direction[:, 1]
                        v_fx = v_final[:, 0]
                        v_fy = v_final[:, 1]
                        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
                        ax.quiver(X, Y, v_x, v_y, angles='xy', scale_units='xy', scale=1, color='olivedrab')
                        ax.quiver(X, Y, v_fx, v_fy, angles='xy', scale_units='xy', scale=1, color='steelblue')
                        ax.set_xlim([-25, 15])
                        # ax.set_ylim([-25, 20])
                        plt.axis('equal')
                        # plt.show()
                else:
                    position = np.array(position)
                    direction_intend = goal_position - position
                    n_intend = direction_intend / (np.linalg.norm(direction_intend, axis=-1)[:, np.newaxis] + 0.00001)
                    v0 = n_intend * np.linalg.norm(v0, axis=-1)[:, np.newaxis]
                    if 0:
                        import matplotlib.pyplot as plt
                        v0_direction = v0 / (np.linalg.norm(v0, axis=-1)[:, np.newaxis] + 0.00001)
                        # plt.figure()
                        ax = plt.gca()
                        X = position[:, 0]
                        Y = position[:, 1]
                        v_x = v0_direction[:, 0]
                        v_y = v0_direction[:, 1]
                        U = direction_intend[:, 0]
                        V = direction_intend[:, 1]
                        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
                        ax.quiver(X, Y, v_x, v_y, angles='xy', scale_units='xy', scale=1, color='olivedrab')
                        ax.set_xlim([-25, 35])
                        ax.set_ylim([-25, 35])
                        # plt.axis('equal')
                        # plt.show()
        return np.array(pre_position)  # (prediction_horizon,ped,2)

class Predictor_zan(Predictor_CVM):
    def __init__(self, dataset, method_cfg, parameters=['default']):
        Predictor_CVM.__init__(self, dataset, method_cfg)

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'm']):
            self.m = method_cfg['param']['optimal'][parameters[1]]['m']
        else:
            self.m = method_cfg['param']['default']['m']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'tau']):
            self.tau = method_cfg['param']['optimal'][parameters[1]]['tau']
        else:
            self.tau = method_cfg['param']['default']['tau']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'r_i']):
            self.r_i = method_cfg['param']['optimal'][parameters[1]]['r_i']
        else:
            self.r_i = method_cfg['param']['default']['r_i']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'r_ij']):
            self.r_ij = method_cfg['param']['optimal'][parameters[1]]['r_ij']
        else:
            self.r_ij = method_cfg['param']['default']['r_ij']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'K']):
            self.K = method_cfg['param']['optimal'][parameters[1]]['K']
        else:
            self.K = method_cfg['param']['default']['K']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'Ko']):
            self.Ko = method_cfg['param']['optimal'][parameters[1]]['Ko']
        else:
            self.Ko = method_cfg['param']['default']['Ko']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'sf_anisotropic_lambda']):
            self.sf_anisotropic_lambda = method_cfg['param']['optimal'][parameters[1]]['sf_anisotropic_lambda']
        else:
            self.sf_anisotropic_lambda = method_cfg['param']['default']['sf_anisotropic_lambda']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'future_horizon']):
            self.future_horizon = method_cfg['param']['optimal'][parameters[1]]['future_horizon']
        else:
            self.future_horizon = method_cfg['param']['default']['future_horizon']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'a_z']):
            self.a_z = method_cfg['param']['optimal'][parameters[1]]['a_z']
        else:
            self.a_z = method_cfg['param']['default']['a_z']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'b_z']):
            self.b_z = method_cfg['param']['optimal'][parameters[1]]['b_z']
        else:
            self.b_z = method_cfg['param']['default']['b_z']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'a_zo']):
            self.a_zo = method_cfg['param']['optimal'][parameters[1]]['a_zo']
        else:
            self.a_zo = method_cfg['param']['default']['a_zo']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'b_zo']):
            self.b_zo = method_cfg['param']['optimal'][parameters[1]]['b_zo']
        else:
            self.b_zo = method_cfg['param']['default']['b_zo']

    def calculate_fio_zan(self, position_i, velocity_i, p_min, r_io):
        # limit the minimum value of t_io, so can prevent sharpe increase of f_io
        v_i = np.linalg.norm(velocity_i)
        n_i = velocity_i / (v_i + 0.000000001)
        destination_io = p_min - position_i  # vector from i point to j
        velocity_io = velocity_i
        v_io = np.linalg.norm(velocity_io)
        d_io = np.linalg.norm(destination_io)
        cos_theta_io = np.dot(velocity_io, destination_io) / ((v_io * d_io) + 0.000000001)
        if cos_theta_io < math.cos(math.pi / 3):
            t_io = math.inf
        else:
            if d_io < r_io:  # still need to consider about it
                t_io = 0.2
            else:
                t_io = cos_theta_io * (d_io - r_io) / v_io  # here add r_ij into consideration
                if t_io < 0:
                    t_io = math.inf
                elif t_io < 0.2:
                    t_io = 0.2
                elif t_io > self.future_horizon * self.delta_T:  # consider in finite horizon
                    t_io = self.future_horizon * self.delta_T
        # if t_io == 0.2:
        #     print('oops!')
        if t_io == math.inf:
            f_io = 0
            f_io_phy = 0
        else:
            destination_vector_prime = position_i + velocity_i * t_io - p_min
            d_io_prime = np.linalg.norm(destination_vector_prime)
            n_io_prime = destination_vector_prime / d_io_prime  # the direction of f-ij-prime
            cos_fai_io = np.dot(n_i, n_io_prime)  #### Q: here is nij or nij prime???
            w = self.sf_anisotropic_lambda + (1 - self.sf_anisotropic_lambda) * (1 + cos_fai_io) / 2
            # f_io = w * (2 * self.a_z * v_i / t_io * np.exp(-abs(r_io - d_io_prime) / (self.b_z * 2))
            #             + self.K * g(r_io - d_io)) * n_io_prime
            # f_io = w * (self.a_zo * v_i / t_io * np.exp(-abs(r_io - d_io_prime) / self.b_zo) +
            #             self.Ko * g(r_io - d_io_prime)) * n_io_prime
            # f_io = w * (self.a_zo * v_i / t_io * np.exp(-abs(r_io - d_io_prime) / self.b_zo)) * n_io_prime
            f_io = w * (self.a_zo * v_i / (1 * t_io) * np.exp(-abs(r_io - d_io_prime) / self.b_zo)) * n_io_prime
            f_io_phy = self.Ko * g(r_io - d_io_prime) * n_io_prime
            # compare with cpp, there is no rij
        return f_io, f_io_phy

    def predict_zan_sf(self, position, velocity, v_0):
        new_velocity = []
        new_position = []
        for i in range(len(position)):
            # v_0 = velocity[i]  # velocity[i]= [vx,vy]
            v_i = np.linalg.norm(velocity[i])
            n_i = velocity[i] / (v_i + 0.000001)
            f_i = self.m * 1 / self.tau * (v_0[i] - velocity[i])  # now v0 = v[i]
            f_ij_sum = []
            f_ij_phy_sum = []
            t_ij_sum = []
            theta_j_sum = np.zeros(len(position))
            for j in range(len(position)):  # calculate f_ij
                if j != i:
                    # calculate cos_theta_ij
                    destination_ij = position[j] - position[i]  # vector from i point to j
                    velocity_ij = velocity[i] - velocity[j]
                    v_ij = np.linalg.norm(velocity_ij)
                    d_ij = np.linalg.norm(destination_ij)
                    cos_theta_ij = np.dot(velocity_ij, destination_ij) / ((v_ij * d_ij) + 0.000000001)
                    theta_j_sum[j] = cos_theta_ij
                    if cos_theta_ij < math.cos(math.pi / 4):
                        t_ij = math.inf
                    elif d_ij < self.r_ij:
                        t_ij = 0.2
                    else:
                        t_ij = cos_theta_ij * (d_ij - self.r_ij) / v_ij  # here add r_ij into consideration
                        if t_ij < 0:
                            t_ij = math.inf
                        elif t_ij < 0.2:
                            t_ij = 0.2
                        elif t_ij > self.future_horizon * self.delta_T:  # consider in finite horizon
                            t_ij = self.future_horizon * self.delta_T
                    t_ij_sum.append(t_ij)
            # if t_ij_sum is empty
            if len(t_ij_sum):
                pass
            else:
                # t_ij_sum = [0]      # 0306 change to inf, but not yet check in simulation
                t_ij_sum = [math.inf]
            t_i = min(t_ij_sum)  # an important consideration is: what if t_i is inf?
            for j in range(len(position)):  # calculate f_ij
                if j != i:
                    if t_i == math.inf:
                        f_ij = 0
                        f_ij_phy = 0
                    else:
                        destination_vector_prime = position[i] + velocity[i] * t_i - (position[j] + velocity[j] * t_i)
                        d_ij_prime = np.linalg.norm(destination_vector_prime)
                        n_ij_prime = destination_vector_prime / d_ij_prime  # the direction of f-ij-prime
                        cos_fai_ij = np.dot(n_i, n_ij_prime)  #### Q: here is nij or nij prime???
                        w = self.sf_anisotropic_lambda + (1 - self.sf_anisotropic_lambda) * (1 + cos_fai_ij) / 2
                        f_ij = w * (self.a_z * v_i / t_i * np.exp(-abs(self.r_ij - d_ij_prime) / self.b_z)) * n_ij_prime
                        # f_ij = w * (self.a_z * v_i / t_i * np.exp(-abs(self.r_ij - d_ij_prime) / self.b_z)+self.K * g(self.r_ij - d_ij_prime)) * n_ij_prime
                        f_ij_phy = self.K * g(self.r_ij - d_ij_prime) * n_ij_prime
                        # compare with cpp, there is no rij
                        # f_ij = w * (a_z * v_i / t_i * np.exp(-abs(d_ij)/b_z)) * n_ij_prime
                    f_ij_sum.append(f_ij)
                    f_ij_phy_sum.append(f_ij_phy)
            f_ij_total = np.sum(f_ij_sum, axis=0) + np.sum(f_ij_phy_sum, axis=0)
            if self.obstacle_type == 1:
                d = np.sqrt(np.sum((position[i] - self.world_d2.T) ** 2, axis=1))
                d_min_index = np.argmin(d)
                p_min = self.world_d2[:, d_min_index]
                r_io = self.r_ij
                f_io, f_io_phy = self.calculate_fio_zan(position[i], velocity[i], p_min, r_io)
            elif self.obstacle_type == 2:
                pos_obs = np.array([e[0] for e in self.world_d2]) # positions of the obstacles
                d = np.sqrt(np.sum((position[i] - pos_obs) ** 2, axis=1))
                d_min_index = np.argmin(d)
                near_ob = self.world_d2[d_min_index][0]
                r = self.world_d2[d_min_index][1] # radius of the first obstacle
                r_io = self.r_i + r
                # only consider the nearest obstacle
                f_io, f_io_phy = self.calculate_fio_zan(position[i], velocity[i], near_ob, r_io)  # k=0.15
                f_io = 2 * f_io
                # f_io_all.append(f_io)  # for debug
            else:
                f_io = 0  # calculate the repulsive force from obstacles
                f_io_phy = 0
            f = f_i + f_ij_total + f_io + f_io_phy
            new_velocity_i = velocity[i] + f / self.m * self.delta_T
            # should give a max acceleration
            new_velocity_i = limit_acceleration(new_velocity_i, v_0[i])
            new_position_i = position[i] + new_velocity_i * self.delta_T
            new_velocity.append(new_velocity_i)
            new_position.append(new_position_i)
        return new_position, new_velocity

    def predict_single(self, scenario):
        frame_tra = scenario.trajectories
        position = []  # the position of last history timestep, if 2 trajectories, 2 lists, each list is ndarray (2,)
        velocity = []
        v0 = []
        goal_position = self.get_goal_short(frame_tra)
        goal_position_l = copy.deepcopy(goal_position)
        if self.load_goal_file:
            goal_in_frame = get_goal(frame_tra, self.goal_list)
            for i in range(len(goal_in_frame)):
                if goal_in_frame[i] != 10:
                    goal_position_l[i] = self.goal_list[goal_in_frame[i], :]
        for i in range(len(frame_tra)):  # predict for i-ped in this frame
            # if (len(frame_tra[i]) >= 2):  # choose trajectory of ped[i] in this frame(most is [])
            old_velocity = (frame_tra[i][-1, :] - frame_tra[i][-2, :]) / self.delta_T
            old_position = frame_tra[i][-1, :]
            mean_velocity = self.average_velocity(frame_tra[i])
            position.append(old_position)
            velocity.append(old_velocity)
            v0.append(mean_velocity)
        pre_position = []
        pre_velocity = []
        v0 = np.squeeze(v0, axis=1)  # v0 is intended velocity, ndarray(# of ped, 2)
        velocity = v0
        for _ in range(scenario.prediction_horizon):  # generate new position for each trajectory of ped
            new_position, new_velocity = self.predict_zan_sf(position, velocity, v0)
            if self.uncertainty:
                noise = np.random.normal(0, self.noise_scale / self.frame_frequency, (len(new_position), 2))
                new_position = new_position + noise
            pre_position.append(new_position)
            pre_velocity.append(new_velocity)
            position, velocity = new_position, new_velocity
            if self.consider_goals:
                if self.load_goal_file:
                    position = np.array(position)
                    direction_intend_l = goal_position_l - position
                    direction_intend_d = np.linalg.norm(direction_intend_l, axis=-1)[:, np.newaxis]
                    n_intend = direction_intend_l / (direction_intend_d ** 2 + 0.00001)
                    v0_direction = v0 / (np.linalg.norm(v0, axis=-1)[:, np.newaxis] + 0.00001)
                    v_final = v0_direction + n_intend
                    v_final_direction = v_final / (np.linalg.norm(v_final, axis=-1)[:, np.newaxis] + 0.00001)
                    v0 = v_final_direction * np.linalg.norm(v0, axis=-1)[:, np.newaxis]
                else:
                    position = np.array(position)
                    direction_intend = goal_position - position
                    n_intend = direction_intend / (np.linalg.norm(direction_intend, axis=-1)[:, np.newaxis] + 0.00001)
                    v0 = n_intend * np.linalg.norm(v0, axis=-1)[:, np.newaxis]
                # n_intend = direction_intend.T / (np.linalg.norm(direction_intend, axis=1) + 0.00001)
                # v0 = (n_intend * np.linalg.norm(v0, axis=1)).T
        return np.array(pre_position)  # (prediction_horizon,ped,2)

class Predictor_kara(Predictor_CVM):
    def __init__(self, dataset, method_cfg, parameters=['default']):
        Predictor_CVM.__init__(self, dataset, method_cfg)
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'm']):
            self.m = method_cfg['param']['optimal'][parameters[1]]['m']
        else:
            self.m = method_cfg['param']['default']['m']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'tau']):
            self.tau = method_cfg['param']['optimal'][parameters[1]]['tau']
        else:
            self.tau = method_cfg['param']['default']['tau']

        # considered number of ped
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'N']):
            self.N = method_cfg['param']['optimal'][parameters[1]]['N']
        else:
            self.N = method_cfg['param']['default']['N']

        # safe distance of i
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'safe_di']):
            self.safe_di = method_cfg['param']['optimal'][parameters[1]]['safe_di']
        else:
            self.safe_di = method_cfg['param']['default']['safe_di']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'r_i']):
            self.r_i = method_cfg['param']['optimal'][parameters[1]]['r_i']
        else:
            self.r_i = method_cfg['param']['default']['r_i']

        self.r_j = self.r_i  # radius of j

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'r_ij']):
            self.r_ij = method_cfg['param']['optimal'][parameters[1]]['r_ij']
        else:
            self.r_ij = method_cfg['param']['default']['r_ij']
        
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'future_horizon']):
            self.future_horizon = method_cfg['param']['optimal'][parameters[1]]['future_horizon']
        else:
            self.future_horizon = method_cfg['param']['default']['future_horizon']

        self.t_alpha = self.future_horizon * self.delta_T  # safe time

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'k']):
            self.k = method_cfg['param']['optimal'][parameters[1]]['k']
        else:
            self.k = method_cfg['param']['default']['k']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'd_min']):
            self.d_min = method_cfg['param']['optimal'][parameters[1]]['d_min']
        else:
            self.d_min = method_cfg['param']['default']['d_min']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'd_mid']):
            self.d_mid = self.d_min + method_cfg['param']['optimal'][parameters[1]]['d_mid']
        else:
            self.d_mid = self.d_min + method_cfg['param']['default']['d_mid']

        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'd_max']):
            self.d_max = self.d_mid + method_cfg['param']['optimal'][parameters[1]]['d_max']
        else:
            self.d_max = self.d_mid + method_cfg['param']['default']['d_max']
        
        # scale parameter for obstacles
        if parameters[0]=='optimal' and exists(method_cfg, ['param', 'optimal', parameters[1], 'kara_scale']):
            self.kara_scale = method_cfg['param']['optimal'][parameters[1]]['kara_scale']
        else:
            self.kara_scale = method_cfg['param']['default']['kara_scale']

    def get_time(self, v, p_j, p_i, r):  # consider about v = 0
        d_ij = p_j - p_i
        a = np.dot(v, v)
        b = -2 * np.dot(d_ij, v)
        c = np.dot(d_ij, d_ij) - (self.safe_di + r) ** 2
        if a == 0:
            return []
        elif (b * b - 4 * a * c) < 0:
            return []
        else:
            sol1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            sol2 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return [sol1, sol2]

    def calculate_fe(self, D):
        a1 = self.k / (self.d_min * (self.d_mid - self.d_max))
        b1 = -a1 * self.d_max
        if D < self.d_min:
            f_e = self.k / D  # self-defined function
        elif D < self.d_mid:
            f_e = self.k / self.d_min
        elif D < self.d_max:
            f_e = a1 * D + b1
        else:
            f_e = 0
        return f_e

    def calculate_fio_kara(self, position_i, v_desired, p_min, r_o):
        sol = self.get_time(v_desired, p_min, position_i, r_o)
        if len(sol) == 0:
            sol = [math.inf, math.inf]
        t1 = sol[0]
        t2 = sol[1]
        if (t1 > 0) & (t1 < self.t_alpha):
            tc = t1
        elif (t1 < 0) & (t2 > 0):
            tc = 0.1
        else:
            tc = math.inf
        if tc == math.inf:
            f_io = 0
        else:
            c_i = position_i + tc * v_desired
            c_o = p_min
            n_io = (c_i - c_o) / (np.linalg.norm(c_i - c_o) + 0.000001)
            D = abs(np.linalg.norm(c_i - position_i) + (np.linalg.norm(c_i - c_o) - self.r_i))  # avoid negative
            f_mag = self.kara_scale * self.calculate_fe(D)  # f is piecewise function
            f_io = f_mag * n_io
        return f_io

    # obstacle not correct
    def predict_kara_sf(self, position, velocity, v_0):
        new_velocity = []
        new_position = []

        for i in range(len(position)):
            # v_0 = velocity[i]  # velocity[i]= [vx,vy]
            # n_i = velocity[i] / (np.linalg.norm(velocity[i]) + 0.000001)
            f_i = self.m * 1 / self.tau * (v_0[i] - velocity[i])  # now v0 = v[i]
            v_desired = velocity[i] + f_i / self.m * self.delta_T
            tc = np.ones(len(position)) * math.inf
            for j in range(len(position)):  # calculate f_ij
                if j != i:
                    relative_v = v_desired - velocity[j]
                    sol = self.get_time(relative_v, position[j], position[i], self.r_j)
                    if len(sol) == 0:
                        sol = [math.inf, math.inf]
                    t1 = sol[0]
                    t2 = sol[1]
                    if (t1 > 0) & (t1 < self.t_alpha):
                        tc[j] = t1
                    elif (t1 < 0) & (t2 > 0):
                        tc[j] = 0  # in paper, here is tc_ij =  0, we tried 0.1
            sorted_index = np.argsort(tc)
            if len(sorted_index) > self.N:
                sorted_index = sorted_index[0:self.N]  # only takes N smallest tc_ij
            f_e = []
            f_phy_sum = []
            # n = 0
            for k in sorted_index:  # !!! need to consider inf!!!
                tc_ij = tc[k]
                if tc_ij == math.inf:
                    break
                c_i = position[i] + tc_ij * v_desired
                c_j = position[k] + tc_ij * velocity[k]
                n_ij = (c_i - c_j) / (np.linalg.norm(c_i - c_j) + 0.000001)
                # f_phy = 100 * g(np.linalg.norm(c_i - c_j)) * n_ij
                D = abs(np.linalg.norm(c_i - position[i]) + (np.linalg.norm(c_i - c_j) - self.r_ij))  # avoid negative
                f_mag = self.calculate_fe(D)  # f is piecewise function
                f_v = f_mag * n_ij
                f_e.append(f_v)
                # f_phy_sum.append(f_phy)
            w = self.get_w(f_e) # used to be: self.get_w(f_e, 1, 'gaussian')
            w1 = w[::-1]
            f_e_total = np.dot(w1, f_e)
            # f_phy_total = np.dot(w1, f_phy_sum)
            # f_e = f_e + f_v
            # n = n + 1
            if self.obstacle_type == 1:
                d = np.sqrt(np.sum((position[i] - self.world_d2.T) ** 2, axis=1))
                d_min_index = np.argmin(d)  # find the nearest point in obstacles
                p_min = self.world_d2[:, d_min_index]
                f_io = self.calculate_fio_kara(position[i], v_desired, p_min, self.r_j)
            elif self.obstacle_type == 2:
                pos_obs = np.array([e[0] for e in self.world_d2]) # positions of the obstacles
                d = np.sqrt(np.sum((position[i] - pos_obs) ** 2, axis=1))
                d_min_index = np.argmin(d)
                near_ob = self.world_d2[d_min_index][0] # position of the closest obstacle
                r = self.world_d2[d_min_index][1] # radius of the closest obstacle
                r_io = self.r_i + r
                # only consider the nearest obstacle
                f_io = self.calculate_fio_kara(position[i], v_desired, near_ob, r - 0.4)
            else:
                f_io = 0  # calculate the repulsive force from obstacles
            f = f_io + f_i + f_e_total  # + f_phy_total
            new_velocity_i = velocity[i] + f / self.m * self.delta_T
            new_velocity_i = limit_acceleration(new_velocity_i, v_0[i])
            new_position_i = position[i] + new_velocity_i * self.delta_T
            new_velocity.append(new_velocity_i)
            new_position.append(new_position_i)
        return new_position, new_velocity

    def predict_single(self, scenario):
        frame_tra = scenario.trajectories
        position = []  # the position of last history time step, if 2 trajectories, 2 lists, each list is ndarray (2,)
        velocity = []
        v0 = []
        goal_position = self.get_goal_short(frame_tra)
        goal_position_l = copy.deepcopy(goal_position)
        if self.load_goal_file:
            goal_in_frame = get_goal(frame_tra, self.goal_list)
            for i in range(len(goal_in_frame)):
                if goal_in_frame[i] != 10:
                    goal_position_l[i] = self.goal_list[goal_in_frame[i], :]

        for i in range(len(frame_tra)):  # predict for i-ped in this frame
            # if (len(frame_tra[i]) >= 2):  # choose trajectory of ped[i] in this frame(most is [])
            old_velocity = (frame_tra[i][-1, :] - frame_tra[i][-2, :]) / self.delta_T
            old_position = frame_tra[i][-1, :]
            mean_velocity = self.average_velocity(frame_tra[i])
            position.append(old_position)
            velocity.append(old_velocity)
            v0.append(mean_velocity)
        pre_position = []
        pre_velocity = []
        v0 = np.squeeze(v0, axis=1)  # v0 is intended velocity, ndarray(# of ped, 2)
        velocity = v0
        for _ in range(scenario.prediction_horizon):  # generate new position for each trajectory of ped
            if self.consider_goals:
                if self.load_goal_file:
                    position = np.array(position)
                    direction_intend_l = goal_position_l - position
                    direction_intend_d = np.linalg.norm(direction_intend_l, axis=-1)[:, np.newaxis]
                    # get the goal vector which is inversely proportional to the distance to the goal position
                    n_intend = direction_intend_l / (direction_intend_d ** 2 + 0.00001)
                    v0_direction = v0 / (np.linalg.norm(v0, axis=-1)[:, np.newaxis] + 0.00001)
                    v_final = v0_direction + n_intend
                    v_final_direction = v_final / (np.linalg.norm(v_final, axis=-1)[:, np.newaxis] + 0.00001)
                    v0 = v_final_direction * np.linalg.norm(v0, axis=-1)[:, np.newaxis]
                else:
                    position = np.array(position)
                    direction_intend = goal_position - position
                    n_intend = direction_intend / (np.linalg.norm(direction_intend, axis=-1)[:, np.newaxis] + 0.00001)
                    v0 = n_intend * np.linalg.norm(v0, axis=-1)[:, np.newaxis]
                # n_intend = direction_intend.T / (np.linalg.norm(direction_intend, axis=1) + 0.00001)
                # v0 = (n_intend * np.linalg.norm(v0, axis=1)).T
            new_position, new_velocity = self.predict_kara_sf(position, velocity, v0)
            if self.uncertainty:
                noise = np.random.normal(0, self.noise_scale / self.frame_frequency, (len(new_position), 2))
                new_position = new_position + noise
            pre_position.append(new_position)
            pre_velocity.append(new_velocity)
            position, velocity = new_position, new_velocity
        return np.array(pre_position)  # (prediction_horizon,ped,2)