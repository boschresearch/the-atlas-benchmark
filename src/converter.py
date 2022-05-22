# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0

# System imports

from unicodedata import name
import pathlib
import yaml

def convert_detection_data_to_json(data, filename):
    with open(filename, 'w') as f:
        for row in data:
            string = '{"track": {"f": ' + str(row[0]) + ', "p": ' + str(row[1]) + ', "x": ' + str(row[2]) + ', "y": ' + str(row[3]) + '}}'
            f.write(string + '\n')

def convert_benchmark_cfg(benchmark_cfg, filename_benchmark_yaml):
     with open(filename_benchmark_yaml, 'w') as file:
            yaml.dump(benchmark_cfg, file, default_flow_style=False)

if __name__ == "__main__":
    pathlib.Path('dataset/newdataset/').mkdir(parents=True, exist_ok=True)
    filename_people = 'dataset/newdataset/detections.ndjson'
    filename_benchmark_yaml = 'cfg/dataset_config_newdataset.yaml'

    # data to be converted into the ndjson file
    # Row format: [frame, person_id, x, y]
    data = [[50, 8, 0.00,4.00],
            [50,13, 1.00,4.00],
            [54, 8, 0.00,4.25],
            [54,13, 1.00,4.25],
            [58, 8, 0.00,4.5],
            [58,13, 1.00,4.5],
            [62, 8, 0.00,4.75],
            [62,13, 1.00,4.75],
            [62,14,-1.00,4.00],
            [66, 8, 0.00,5.00],
            [66,13, 1.00,5.00],
            [66,14,-1.00,3.75],
            [70, 8, 0.00,5.25],
            [70,13, 1.00,5.25],
            [70,14,-1.00,3.50],
            [74, 8, 0.00,5.5],
            [74,13, 1.00,5.5],
            [74,14,-1.00,3.25],
            [78, 8, 0.00,5.75],
            [78,13, 1.00,5.75],
            [78,14,-1.00,3.0],
            [82, 8, 0.00,6.00],
            [82,13, 1.00,6.00],
            [82,14,-1.00,2.74],
            [86, 8, 0.00,6.25],
            [86,13, 1.00,6.25],
            [86,14,-1.00,2.5],
            [90, 8, 0.00,6.5],
            [90,13, 1.00,6.5],
            [90,14,-1.00,2.25]]
    convert_detection_data_to_json(data, filename_people)

    # Necessary inputs to the benchmark config file
    benchmark_cfg = dict()
    benchmark_cfg['dataset'] = dict()
    benchmark_cfg['dataset']['name'] = 'newdataset'
    benchmark_cfg['dataset']['data'] = [filename_people]
    benchmark_cfg['dataset']['frequency'] = 4
    benchmark_cfg['dataset']['annotation interval'] = 4
    benchmark_cfg['dataset']['goals'] = None
    benchmark_cfg['dataset']['map'] = None
    benchmark_cfg['benchmark'] = dict()
    benchmark_cfg['benchmark']['setup'] = dict()
    benchmark_cfg['benchmark']['setup']['interpolate'] = False
    benchmark_cfg['benchmark']['setup']['smooth'] = 0
    benchmark_cfg['benchmark']['setup']['downsample'] = True
    benchmark_cfg['benchmark']['setup']['downsample rate'] = 2
    benchmark_cfg['benchmark']['setup']['downsample map'] = False
    benchmark_cfg['benchmark']['setup']['downsample map rate'] = 1
    benchmark_cfg['benchmark']['setup']['added_noise_sigma'] = 0.0
    benchmark_cfg['benchmark']['setup']['observation period'] = 3
    benchmark_cfg['benchmark']['setup']['prediction horizon'] = 3

    convert_benchmark_cfg(benchmark_cfg, filename_benchmark_yaml)