# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0


from evaluator import Evaluator
import numpy as np

class Benchmark:
    def __init__(self, verbose):
        self.verbose = verbose

    def accuracy_experiment(self, valid_scenes, predictor, metric='ade'):
        if metric not in ['ade','fde', 'kade', 'kfde']:
            print('[ERROR] Benchmark: The metric', metric, 'is not recognized! Supported metrics are \'ade\', \'fde\',\'kade\',\'kfde\'.')
        evaluation = Evaluator(set_GT_length=valid_scenes[0].prediction_horizon)
        results = []
        for scene in valid_scenes:
            predictions = predictor.predict(scene)
            if metric in ['ade','fde']:
                metric_values_ade = evaluation.evaluate_scenario_ade_fde(scene,predictions)
                results.append(metric_values_ade)
            if metric in ['kade','kfde']:
                metric_values_kade = evaluation.evaluate_scenario_kade_kfde(scene,predictions)
                results.append(metric_values_kade)

        if self.verbose:
            if metric in ['ade','fde']:
                print('The mean ADE is', np.mean([result[0] for result in results]), '+-', np.std([result[0] for result in results]))
                print('The mean FDE is', np.mean([result[1] for result in results]), '+-', np.std([result[1] for result in results]))
            if metric in ['kade','kfde']: 
                print('The mean kADE is', np.mean([result[0] for result in results]), '+-', np.std([result[0] for result in results]))
                print('The mean kFDE is', np.mean([result[1] for result in results]), '+-', np.std([result[1] for result in results]))

        if metric in ['ade','kade']:
            result = np.mean([result[0] for result in results])
        if metric in ['fde','kfde']:
            result = np.mean([result[1] for result in results])
            
        return result
