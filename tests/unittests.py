# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0



import sys
sys.path.insert(0,'')

import unittest
from context import Dataset, Scenario, Evaluator
from context import Predictor_sof, Predictor_CVM
from context import Prediction
import yaml
import numpy as np

from context import ROOT_DIR

class TestBehcmmarkMethods(unittest.TestCase):

    def test_dataset_import(self):
        datasets = ['atc', 'eth', 'hotel', 'test_point_obstacles', 'test_traj', 'thor1', 'thor3']
        for name in datasets:
            dataset = self.import_dataset(name)
            self.assertEqual(dataset.name, name)

    def import_dataset(self, name):
        input_dataset = ROOT_DIR + f'/cfg/dataset_config_{name}.yaml'
        with open(input_dataset, 'r') as file:
            benchmark_cfg = yaml.load(file, Loader=yaml.FullLoader)
        dataset = Dataset(benchmark_cfg)
        return dataset

    def test_prediction_in_grid_map(self):
        dataset = self.import_dataset('test_traj')
        
        observation_len = 6
        prediction_horizon = 3
        valid_scenes = dataset.extract_scenarios(prediction_horizon, observation_len)

        # Test Predictor_CVM
        with open('cfg/method_config_cvm.yaml', 'r') as file:
            method_cfg = yaml.load(file, Loader=yaml.FullLoader)
        method_cfg['param']['meta']['load_goal'] = False
        method_cfg['param']['uncertainty']['uncertainty'] = False

        predictor = Predictor_CVM(dataset, method_cfg)
        predictions = predictor.predict(valid_scenes[32])
        prediction_gt = np.array([[[[ 6.25, 10.        ],
            [ 6.25      ,  8.        ],
            [ 3.56531754,  3.80870661]],

            [[ 6.       ,  10.        ],
            [ 6.        ,  8.        ],
            [ 3.47263509,  4.10541322]],

            [[ 5.75     ,  10.        ],
            [ 5.75      ,  8.        ],
            [ 3.37995263,  4.40211984]]]])
        np.testing.assert_array_almost_equal(predictions.trajectories, prediction_gt, decimal=2)

        # Test Predictor_sof
        with open('cfg/method_config_sof.yaml', 'r') as file:
            method_cfg = yaml.load(file, Loader=yaml.FullLoader)
        method_cfg['param']['meta']['load_goal'] = False
        method_cfg['param']['uncertainty']['uncertainty'] = False

        predictor = Predictor_sof(dataset, method_cfg)
        predictions = predictor.predict(valid_scenes[32])
        prediction_gt = np.array([[[[ 6.25067904, 10.00774943],
                                    [ 6.2513521 ,  7.99593593],
                                    [ 3.56336148,  3.80517187]],

                                    [[ 6.00174829, 10.01952913],
                                    [ 6.00350349,  7.99000172],
                                    [ 3.46757221,  4.09628361]],

                                    [[ 5.75306199, 10.03346008],
                                    [ 5.756194  ,  7.98332546],
                                    [ 3.37102044,  4.38605525]]]])
        np.testing.assert_array_almost_equal(predictions.trajectories, prediction_gt, decimal=2)

    def test_prediction_in_point_obstacle_map(self):
        dataset = self.import_dataset('test_point_obstacles')
        
        observation_len = 6
        prediction_horizon = 3
        valid_scenes = dataset.extract_scenarios(prediction_horizon, observation_len)

        # Test Predictor_CVM
        with open('cfg/method_config_cvm.yaml', 'r') as file:
            method_cfg = yaml.load(file, Loader=yaml.FullLoader)
        method_cfg['param']['meta']['load_goal'] = False
        method_cfg['param']['uncertainty']['uncertainty'] = False

        predictor = Predictor_CVM(dataset, method_cfg)
        predictions = predictor.predict(valid_scenes[0])
        prediction_gt = np.array([[[[ 1.5 , 12.1 ]],
            [[ 1.75, 12.1 ]],
            [[ 2.  , 12.1 ]]]])
        np.testing.assert_array_almost_equal(predictions.trajectories, prediction_gt, decimal=2)

        # Test Predictor_sof
        with open('cfg/method_config_sof.yaml', 'r') as file:
            method_cfg = yaml.load(file, Loader=yaml.FullLoader)
        method_cfg['param']['meta']['load_goal'] = False
        method_cfg['param']['uncertainty']['uncertainty'] = False

        predictor = Predictor_sof(dataset, method_cfg)
        predictions = predictor.predict(valid_scenes[32])
        prediction_gt = np.array([[[[ 9.14670775, 12.56724914]],
            [[ 9.312401 , 12.67495912]],
            [[ 9.4763887, 12.78308141]]]])
        np.testing.assert_array_almost_equal(predictions.trajectories, prediction_gt, decimal=2)

    def test_metrics_ade_fde_kade_kfde(self):
        startframe = 0
        trajectories = 0
        pedestrian_ids = 0
        gt = [np.array( [[0,0],[1,1],[2,2],[3,3]] ), np.array( [[10,10],[11,11],[12,12],[13,13]] )]

        observation_length = 0
        prediction_horizon = 4
        evaluation = Evaluator(set_GT_length=prediction_horizon)

        s = Scenario(startframe,trajectories, pedestrian_ids, gt, prediction_horizon, observation_length)

        prediction = Prediction([np.array([[[0,0.1],[10,10.1]],[[1,1],[11,11]],[[1.9,2.1],[11.9,12.1]],[[3.1,3],[13.1,13]]]),
                                np.array([[[0,0],[10,10]],[[1,2],[11,12]],[[3,4],[13,14]],[[5,6],[15,16]]]),
                                np.array([[[0,-1],[10,9]],[[1,2],[11,12]],[[2,3],[12,13]],[[5,4],[15,14]]])])
        ade, fde = evaluation.evaluate_scenario_ade_fde(s,prediction)
        self.assertAlmostEqual(ade, 1.0349257155584066, places=7)
        self.assertAlmostEqual(fde, 1.9805397509879263, places=7)
        kade, kfde = evaluation.evaluate_scenario_kade_kfde(s,prediction)
        self.assertAlmostEqual(kade, 0.0603553390593273, places=7)
        self.assertAlmostEqual(kfde, 0.09999999999999987, places=7)


if __name__ == '__main__':
    unittest.main()
