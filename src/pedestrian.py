# Copyright (c) 2022 - for information on the respective copyright owner see the NOTICE file or the repository https://github.com/boschresearch/the-atlas-benchmark
#
# SPDX-License-Identifier: Apache-2.0


class Pedestrian:
    def __init__(self):
        self.start_frame = []  # first time stamp frame when the pedestrian is detected
        self.end_frame = [] # last time stamp frame when the pedestrian is detected
        self.frames = []  # vector of time stamp frames when the pedestrian is detected
        self.ped_id = [] # ID (number) of the pedestrian
        self.trajectory = [] # trajectory of the pedestrian
