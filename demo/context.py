# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dataset import Dataset
from scenario import Scenario
from predict.predictor import Predictor_CVM
from predict.predict_sof import Predictor_sof, Predictor_zan, Predictor_kara
from evaluator import Evaluator
from prediction import Prediction
from benchmark import Benchmark

from predict.new_predictor_template import Newclass
#from predict.predict_trajectronpp import TrajectronPredictor
#from predict.predict_sgan import SGANPredictor
