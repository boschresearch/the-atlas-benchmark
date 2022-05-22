# The Atlas Benchmark

Tha Atlas Benchmark offers a collection of scripts and functions for evaluating 2D trajectory predictors.

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
Get all the submodules first:

```
git submodule update --init --recursive
```

Please make sure to download the models for S-GAN. Once you cd in the folder sgan, run the following script: 

```
bash scripts/download_models.sh
```

Make sure to install atlas in your local virtual environmnet:

```
python3 -m venv atlas-env
```

or we recommend to use conda

```
conda create -n atlas-env python=3.7
```

Afterwards you can run pip to install the requirements (you may need to install swig via apt)

```
pip install -r requirements.txt
```

You can verify the installation by running tests/unittests.py`.

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

For a list of other open source components included in UUV Simulator, see the
file [open_source_licenses.md](open_source_licenses.md).
