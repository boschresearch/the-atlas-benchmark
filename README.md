# The Atlas Benchmark

The Atlas Benchmark offers a collection of scripts and functions for evaluating 2D trajectory predictors.

Atlas allows automated systematic evaluation and comparison of the built-in, external and new prediction methods on several popular datasets (ETH, ATC and THÖR) using probabilistic and geometric accuracy metrics (ADE, FDE, k-ADE and FDE, NLP).

![Predictions](docs/fig/predictions-thor3.png?raw=true "Predictions")

![The Atlas Benchmark](docs/fig/atlas-design.png?raw=true "The Atlas Benchmark")

Important highlights of Atlas include:

1. Supported import of new datasets (labeled detection streams),

2. Support for contextual cues in the environment,

3. Automated calibration of prediction hyperparameters,

4. Automated parametrized scenario extraction, 

5. Direct interface to the prediction methods.

# Installation and setup

```
git clone -b bare-bones https://github.com/boschresearch/the-atlas-benchmark.git
cd the-atlas-benchmark
git submodule add https://gitsvn-nt.oru.se/tim/the-atlas-benchmark-data.git dataset
python3 -m venv atlas-env
source atlas-env/bin/activate
sudo apt install python3-sklearn swig
pip install -r requirements.txt
python tests/unittests.py
code .
```
Try the notebooks `understanding_data_import.ipynb` and `understanding_prediction.ipynb`

If both notebooks work without issues, the Atlas Benchmark is ready to go.

# How to use it

The functionality of Atlas is fully described and illustrated in [docs/tutorial.md](docs/tutorial.md)

For a quick intro, head over to the `demo/` folder and check out the tutorial notebooks:
```
demo/understanding_data_import.ipynb
demo/understanding_prediction.ipynb
```

# Reference

Further details on the motivation and implementation of Atlas can be found in [the following paper](https://darko-project.eu/wp-content/uploads/papers/2021/SocialNav_WS_RSS_2021_Atlas.pdf):

```
@inproceedings{rudenko2021atlas,
  title={Atlas: a Benchmarking Tool for Human Motion Prediction Algorithms},
  author={Rudenko, Andrey and Huang, Wanting and Palmieri, Luigi and Arras, Kai O and Lilienthal, Achim J},
  booktitle={Robotics: Science and Systems (RSS) Workshop on Social Robot Navigation},
  year={2021}
}
```

# Contact

The Atlas Benchmark is developed and maintained by Andrey Rudenko, Luigi Palmieri and Wanting Huang.

In case of questions and comments, feel free to drop us a line at [andrey.rudenko@bosch.com](andrey.rudenko@bosch.com) and [luigi.palmieri@bosch.com](luigi.palmieri@bosch.com)

# License

the-atlas-benchmark is open-sourced under the Apache-2.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in the-atlas-benchmark, see the
file [open_source_licenses.md](open_source_licenses.md).
