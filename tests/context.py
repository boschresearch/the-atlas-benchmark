# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0



import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.dataset import Dataset
from src.scenario import Scenario
from src.predict.predictor import Predictor_CVM
from src.predict.predict_sof import Predictor_sof, Predictor_zan, Predictor_kara
from src.evaluator import Evaluator
from src.prediction import Prediction

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
