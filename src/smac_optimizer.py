from evaluator import Evaluator
from benchmark import Benchmark
from predict.predict_sof import Predictor_sof, Predictor_zan, Predictor_kara
from dataset import Dataset
import random, yaml
import logging
import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.initial_design.default_configuration_design import DefaultConfiguration
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
import importlib

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SmacOptimizer():
    def __init__(self, input_method_cfg_path, input_benchmark_cfg_path, dataset_train_split=[0,1], min_num_prople=1, iterations=100, metric='ade'):
        self.method_cfg_path = input_method_cfg_path
        self.benchmark_cfg_path = input_benchmark_cfg_path
        self.split = dataset_train_split
        self.iterations = iterations
        self.metric = metric
        self.min_num_prople = min_num_prople

    def optimize(self):
        with open(self.method_cfg_path, 'r') as file:
            self.method_cfg = yaml.load(file, Loader=yaml.FullLoader)

        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace()
        x = []
        for param in self.method_cfg['optim'].keys():
            min_val = self.method_cfg['optim'][param]['min']
            max_val = self.method_cfg['optim'][param]['max']
            def_val = self.method_cfg['param']['default'][param]
            x.append(UniformFloatHyperparameter(param, min_val, max_val, default_value=def_val))
        cs.add_hyperparameters(x)  #, x5, x6

        # Scenario object
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                            "runcount-limit": self.iterations,
                            # max. number of function evaluations; for this example set to a low number
                            "cs": cs,  # configuration space
                            "deterministic": "true",
                            "output_dir": None
                            })

        with open(self.benchmark_cfg_path, 'r') as file:
            benchmark_cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.observation_len = benchmark_cfg['benchmark']['setup']['observation period']
        self.prediction_horizon = benchmark_cfg['benchmark']['setup']['prediction horizon']

        # Optional split parameter defines the portion of the dataset that is being extracted,
        # frames between 0 * length(dataset) and 1 * length(dataset)
        self.dataset_train = Dataset(benchmark_cfg, self.split)
        self.valid_scenes_train = self.dataset_train.extract_scenarios(self.prediction_horizon,self.observation_len, min_num_prople=self.min_num_prople)

        self.bench = Benchmark(verbose = False)

        # Example call of the function
        # It returns: Status, Cost, Runtime, Additional Infos
        def_value = self.evaluate_config(cs.get_default_configuration())
        print("Default Value: %.2f" % def_value)

        # Optimize, using a SMAC-object
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(42),
                        initial_design=DefaultConfiguration,
                        tae_runner=self.evaluate_config)

        incumbent = smac.optimize()

        # re-opening the cfg file to restore the original default values
        with open(self.method_cfg_path, 'r') as file:
            self.method_cfg = yaml.load(file, Loader=yaml.FullLoader)
        print('The final paramaters are:', incumbent)
        for param in self.method_cfg['optim'].keys():
            # If the dataset name is not in the method config file
            if self.dataset_train.name not in self.method_cfg['param']['optimal'].keys():
                self.method_cfg['param']['optimal'][self.dataset_train.name] = dict()
            self.method_cfg['param']['optimal'][self.dataset_train.name][param] = incumbent[param]
        
        with open(self.method_cfg_path, 'w') as file:
            yaml.dump(self.method_cfg, file, default_flow_style=False)

    def evaluate_config(self,x):
        # load the incumbent optimization parameters
        # str(key) is parameter name
        # x[str(key)] is parameter value
        for key in x:
            self.method_cfg['param']['default'][str(key)] = x[str(key)]

        PredictorClass = getattr(importlib.import_module("context"), self.method_cfg['class'])
        predictor = PredictorClass(self.dataset_train, self.method_cfg)
        result = self.bench.accuracy_experiment(self.valid_scenes_train, predictor, self.metric)
        
        print(result)
        return result