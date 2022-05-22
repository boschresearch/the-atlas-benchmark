# The Atlas Benchmark Tutorials

Tha Atlas Benchmark offers a collection of scripts and functions for evaluating 2D trajectory predictors. This document presents the complete functionality of Atlas.

## Preliminary steps

Before proceeding, make sure to install Atlas as described in [README.md](../README.md). You can verify the installation by navigating in the `tests/` folder and running `unittests.py`.

## 1. Project structure and classes overview

```
the-atlas-benchmark/
-- cfg/
```
stores the config files of the datasets and the prediction methods.
```
-- dataset/
```
stores the dataset files.
```
-- demo/
```
stores the tutorial notebooks.
```
-- src/
```
stores the code files.
```
-- tests/
```
stores the unit tests.
```
-- README.md
-- requirements.txt
```

The Atlas benchmark is based on several functional classes, which guide the process from data import to prediction, visualization and evaluation. The classes files are located in the `src/` folder. In this section a brief description of the classes is provided. Their fields and methods are described in the following tutorials.

- `Dataset` takes the dataset config file as input, stores and pre-processes the detections and the map data.

- `Pedestrian` stores the entire trajectory in the `Dataset` for a single pedestrian. It is used in the `Dataset` class.

- `Scenario` object stores a valid scenario (i.e. without gaps in observation data for all people), defined by the `observation_length` and `prediction_horizon`. It includes a formalized representation of the past trajectories and the ground truth, passed as input to the `Predictor` object.

- `Predictor` is a parent class for the prediction methods. `Predictor_CVM`, `Predictor_kara` and `Predictor_zan` inherit from it. Most importantly, an object of the `Predictor` class includes a .`predict()` function, which takes a `Scenario` object as input and returns a `Prediction` object.

- `Prediction` is a formalized representation of the predicted trajectories, returned by `Predictor`.

- `Evaluator` wrapper includes the evaluation metrics and visualization tools.

- `SmacOptimizer` wrapper allows the optimization of a method's hyperparameters on a given dataset.

- `Benchmark` is a supplementary class, currently only used by the `SmacOptimizer`. it includes an automated accuracy experiment, which returns a single score of the target metric. In order to achieve greater flexibility in results and representation, we opted for jupyter notebook-based experiments instead, described in `demo/understanding_benchmark_experiments.ipynb`.

## 2. Data format and data import

This tutorial is illustrated by the `demo/understanding_data_import.ipynb` notebook, which can be executed for a given dataset (uncomment the desired one in the 2nd cell of the notebook).

### Dataset files

The input detection data in contained in `json` files in the `dataset/` folder (accessed from the third-party repo in the Installation step, see above). The detection file includes one row per detection, which stores the person ID `p`, frame ID `f`, and `x`, `y` positions. Here is an example from the ETH dataset, which includes three frames with two people.

```
{"track": {"f": 3689, "p": 64, "x": 2.74, "y": 3.33}}
{"track": {"f": 3689, "p": 68, "x": 2.06, "y": 2.16}}
{"track": {"f": 3695, "p": 64, "x": 2.09, "y": 3.2}}
{"track": {"f": 3695, "p": 68, "x": 2.8, "y": 2.21}}
{"track": {"f": 3701, "p": 64, "x": 1.41, "y": 2.94}}
{"track": {"f": 3701, "p": 68, "x": 3.36, "y": 2.52}}
```

Similarly, the gridmap of the environment is stored in another `json` file:

```
{"map": {"id": "thor_Ex_1", "origin_x":-11, "origin_y":-10, "theta":0, "h":230, "w":235, "resolution":0.1}}
{"surface": {"x":0, "y":0, "surf_class":1}}
{"surface": {"x":0, "y":1, "surf_class":1}}
{"surface": {"x":0, "y":2, "surf_class":1}}
{"surface": {"x":0, "y":3, "surf_class":1}}
```

This file also includes the reference frame details, i.e. the `origin_x` [m], `origin_y` [m], `theta` and `resolution` [m] fields. 

The surface classes are described in the third `json` as follows:

```
{"surf_class": {"id":0, "color":"w", "name":"free_space"}}
{"surf_class": {"id":1, "color":"k", "name":"obstacle"}}
```

Alternatively, the binary environment with free and occupied space can be accessed from a `.png` image.

Goal positions are stored in a separate file as a list of `x` and `y` coordinates in the map frame, for instance in THÖR:

```
4.440   8.550
9.030   3.580
1.110   -3.447
-5.468  -6.159
-0.130   4.150
```

### Dataset config

The dataset files are referenced by the config file in the `cfg/` folder, which provides the necessary details about the dataset.


- `name`: [str], name of the dataset
- `data`: [list of strings], one or more detection files
- `frequency`: [float], frame rate of the annotated frames
- `annotation interval`: [int], step of annotation, this is necessary for interpolation. For instance, if every consecutive frame is annotated as in THÖR, this value is 1. In the ETH dataset it is 6.
- `goals`: [str], goals file
- `map`:
    - `type`: `pic` or `sem` (semantic)
    - `picture`: [str], path to the map picture
    - `H_matrix`: [str], path to the map transformation matrix (only used in the ETH dataset, otherwise set it to `None`)
    - `sem_class`: [str], path to the file with the semantic classes description
    - `obstacle_type`: 0,1 or 2, the type of obstacles considered: 0 for obstacle-unaware, 1 for grid-map, 2 for point-radius obstacles

### Pre-processing

The following fields in the dataset config file defined the pre-processing steps, which will be applied to the dataset:


- `interpolate`: [bool], 'True' if the missing detections in the dataset should be interpolated
- `smooth`: [int], trajectory smoothing value in the average moving filter
- `downsample`: [bool], 'True' if the trajectories should be downsampled
- `downsample rate`: 1 # fps  1 for no downsampling
- `downsample map`: [float], trajectory downsampling ratio
- `downsample map rate`: [float], map downsampling ratio
- `added_noise_sigma`: [float], sigma value of the added Gaussian white noise
- `observation period`: [int], the number of observation points. This value will be reflected in the scenario generation from the selected dataset
- `prediction horizon`: [int], the number of prediction steps. This value will be reflected in the scenario generation from the selected dataset

Other option for dataset import include:
- Dataset split feature, `split=[a,b]` passed to the constructor, allows extraction of a certain portion of the dataset, `a` and `b` being the interval between 0 and 1.

### Data converter

The benchmark includes a `src/converter.py` for an arbitrary detections dataset, which creates the nessesary files (`cfg` and `json`) for the dataset to be used in Atlas. This converter also serves as a minimum workable example for the dataset configuration in Atlas.

### Included datasets

Atlas includes several popular datasets:
- ETH https://icu.ee.ethz.ch/research/datsets.html
  - dataset_config_eth.yaml
  - dataset_config_hotel.yaml
- ATC https://dil.atr.jp/crest2010_HRI/ATC_dataset/
  - dataset_config_atc.yaml
- THÖR http://thor.oru.se
  - dataset_config_thor1.yaml
  - dataset_config_thor3.yaml

Furthermore, several simulated scenarios of distinct social interactions are available for visually inspecting the behavior of various predictors:
- dataset_config_simulated_chasing.yaml
- dataset_config_simulated_crossing.yaml
- dataset_config_simulated_opposing.yaml

### Summary

The dataset is imported as

```
input_dataset = '../cfg/dataset_config_thor3.yaml'
with open(input_dataset, 'r') as file:
    benchmark_cfg = yaml.load(file, Loader=yaml.FullLoader)
dataset = Dataset(benchmark_cfg, split=[0,0.7])
```

## 3. Scenario generation and prediction

This tutorial is illustrated by the `demo/understanding_prediction.ipynb` notebook, which can be executed for a given dataset (uncomment the desired one in the 2nd cell of the notebook).

Scenarios are extracted from the `dataset` as 

```
valid_scenes = dataset.extract_scenarios(prediction_horizon, observation_len)
```

This returns a list of `Scenario` objects. An optional argument to `extract_scenarios` is `min_num_prople=n` [int], which filters out scenarios with less than n people.

Each scenario has the following main fields:
- `trajectories`: list of numpy arrays containing the observed trajectories
- `gt`: list of nuumpy arrays containing the ground truth future trajectories
- `observation_length` and `prediction_horizon` parameters for this scenario

A scenario is passed to one of the `Predictor` objects via the `.predict()` function, for instance:

```
with open('../cfg/method_config_kara.yaml', 'r') as file:
    method_cfg = yaml.load(file, Loader=yaml.FullLoader)
predictor = Predictor_kara(dataset, method_cfg)
predictions = predictor.predict(scenario)
```

### Prediction parameters

Parameters of each supplied predictor are read from an individual config file, located in the `cfg/` folder. They include both method specific hyperparameters, which can be optimized with the provided SMAC3 interface, and benchmark-relevant uncertainty settings:
- `uncertainty`:
  - `uncertainty`: [bool], 'True' if multiple samples are requested
  - `num_particles`: [int], the number of samples

### Prediction output format

Prediction output has the following structure, considering K sampled trajectories for N people and T time steps:

```
[sample_1[t][p][x,y], sample_2[t][p][x,y], ..., sample_K[t][p][x,y]]
```
where each `sample` is a numpy array, `t` and `p` go from 0 to T-1 and N-1 respectively.

## 4. Evaluation

Evaluation using quantitative metrics, qualitative visualization and animation tools is performed from the `Evaluator` object:

```
evaluation = Evaluator(set_GT_length=prediction_horizon)
```

### Metrics

The following accuracy metrics are available:

- Average and Final Displacement errors (ADE, FDE) calculate the average Euclidean distance between the ground truth position and the predicted position at each time step:
  ```
  evaluation.evaluate_scenario_ade_fde(scenario,predictions)
  ```
  If called for an undertain prediction with K sampled trajectories, this metric calculates the average values among all samples for the corresponding time step.
- kADE and kFDE calculate the average displacement between the ground truth position and the closest predicted sample: 
  ```
  evaluation.evaluate_scenario_kade_kfde(scenario,predictions)
  ```
  ADE and kADE return identical results in case uncertainty is not modeled.
- Negative Log-Likelihood (NLL) estimates the pdf value of the ground truth position in a Gaussian function, fit to the samples of the final position estimation:
  ```
  evaluation.cal_nll(scenario,predictions)
  ```
  This metric only works with uncertain predictions.

### Visualization

Scenario plotting works for both uncertain and certain predictions, as well as for a scenario without predictions (i.e. showing only the ground truth), if the `predictions` argument is not given:
```
evaluation.plot_scenario(dataset, scenario, predictions)
```
Animation is possible as well:
```
ani = evaluation.draw_animation(dataset, scenario, predictions)
ani.save('<path>/prediction_animation.gif', writer='imagemagick')
```

### Included baselines: Model-based predictors

The Constant Velocity Model (CVM), Social Force model (Helbing and Molnar 1995) and its extensions to explicitly model future collisions (Karamouzas et al. 2009, Zanlungo et al. 2011) are included as baselines. See the `Predictor` classes:
- `Predictor_CVM`
- `Predictor_sof`
- `Predictor_kara`
- `Predictor_zan`

Apart from method-specific hyperparameters, these methods have the following common settings in their config files:
- `v0_mode`: [str], initial velocity filtering mode ("gaussian", "linear" or "constant")
- `v0_sigma`: [float], sigma value to determine the initial velocity in the "gaussian" mode
- `set_goal`: [bool], enabling projection of the local navigation goal in the current movement direction of each agent
- `goal_step`: [int], the amount of steps projected ahead of the agent to simulate the local goal

### Included baselines: Learning-based predictors

Interfaces for several popular learning-based predictors are included in Atlas:
- Social GAN (Gupta et al. 2018), https://github.com/agrimgupta92/sgan
  - It is accessed via the `SGANPredictor` wrapper, see the `understanding_prediction_with_sgan.ipynb` notebook
- Trajectron++ (Salzmann et al. 2020), https://github.com/StanfordASL/Trajectron-plus-plus
  - It is accessed via the `TrajectronPredictor` wrapper, see the `understanding_prediction_with_trajectronpp.ipynb` notebook

## 5. Benchmarking

By combining the simple blocks presented above, it is possible to set up the experiments in Atlas.

For instance, accuracy experiment conditioned on the prediction horizon is possible as follows:

```
input_dataset = '../cfg/dataset_config_thor3.yaml'
with open(input_dataset, 'r') as file:
    benchmark_cfg = yaml.load(file, Loader=yaml.FullLoader)
dataset = Dataset(benchmark_cfg, split=[0,0.7])
observation_len = benchmark_cfg['benchmark']['setup']['observation period']

# A loop over the prediction horizon values
for prediction_horizon in [3,5,7,10]
    valid_scenes = dataset.extract_scenarios(prediction_horizon, observation_len)

    predictor_uncertain = Predictor_kara(dataset, method_cfg)

    results_ade = []
    results_kade = []
    for i in range(len(valid_scenes)):
        predictions = predictor_uncertain.predict(valid_scenes[i])
        metric_values_ade = evaluation.evaluate_scenario_ade_fde(valid_scenes[i],predictions)
        metric_values_kade = evaluation.evaluate_scenario_kade_kfde(valid_scenes[i],predictions)
        results_ade.append(metric_values_ade)
        results_kade.append(metric_values_kade)

    print('The mean ADE is', np.mean([result[0] for result in results_ade]), '+-', np.std([result[0] for result in results_ade]))
    print('The mean FDE is', np.mean([result[1] for result in results_ade]), '+-', np.std([result[1] for result in results_ade]))
```

## 6. SMAC3 hyperparameter optimization

This tutorial is illustrated by the `demo/understanding_benchmark_experiments.ipynb` notebook.

The SMAC3 hyperparameter optimization system (Lindauer et al. 2017) is set to automatically read the method parameters from the config file and optimize them on a given dataset.

Which parameters to optimize, as well as their search limits, are defined in the `optim` branch of the config. The inital values are set to the `default` ones, for instance in this example `K`, `a` and `b` are optimized within the given limits:
- optim:
  - K:
    - max: 1200
    - min: 0
  - a:
    - max: 800
    - min: 1
  - b:
    - max: 10
    - min: 0.005
- param:
  - default:
    - K: 215
    - a: 13.4
    - b: 4.15
    - future_horizon: 40
    - m: 80
    - r_ij: 0.5
    - tau: 0.5
    - ...

The optimization runs for a number of `iterations` [int] and the target `metric` ['ade','fde', 'kade', 'kfde']. The training data and scenario extraction parameters are accessed from the dataset config file, it is also possible to set the additional `dataset_train_split` and `min_num_prople` parameters. Optimal values for the given dataset name are written (accordingly, overwritten) to the method config file as:

- param:
  - optimal:
    - \<dataset name\>
      - \<parameter name\> : \<optimal value\>
      - ...


Loading the optimal parameters for the provided predictors is possible with an optional argument:
```
predictor_optimal = Predictor_kara(dataset, method_cfg, parameters=['optimal','eth'])
```

### Summary

The optimization is achieved with

```
input_benchmark_cfg_path = '../cfg/dataset_config_eth.yaml'
input_method_cfg_path = '../cfg/method_config_kara.yaml'
e = SmacOptimizer(input_method_cfg_path, input_benchmark_cfg_path, dataset_train_split=[0,0.2], min_num_prople=2, iterations=300, metric='fde')
e.optimize()
```

# References

D. Helbing and P. Molnar, “Social force model for pedestrian dynamics,” Physical review E, vol. 51, no. 5, p. 4282, 1995.

I. Karamouzas, P. Heil, P. van Beek, and M. H. Overmars, “A predictive collision avoidance model for pedestrian simulation,” in Int. Workshop on Motion in Games. Springer, 2009, pp. 41–52.

F. Zanlungo, T. Ikeda, and T. Kanda. Social force model with explicit
collision prediction. EPL (Europhysics Letters), 93(6):68005, 2011.

A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi, “Social GAN: Socially acceptable trajectories with generative adversarial networks,” in Proc. of the IEEE Conf. on Comp. Vis. and Pat. Rec. (CVPR), June 2018.

T. Salzmann, B. Ivanovic, P. Chakravarty, and M. Pavone, “Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data,” in European Conference on Computer Vision. Springer, 2020, pp. 683–700.

M. Lindauer, K. Eggensperger, M. Feurer, S. Falkner, A. Biedenkapp,
and F. Hutter, “Smac v3: Algorithm configuration in python,” https:
//github.com/automl/SMAC3, 2017.
