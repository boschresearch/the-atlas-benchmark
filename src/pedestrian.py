class Pedestrian:
    def __init__(self):
        self.start_frame = []  # first time stamp frame when the pedestrian is detected
        self.end_frame = [] # last time stamp frame when the pedestrian is detected
        self.frames = []  # vector of time stamp frames when the pedestrian is detected
        self.ped_id = [] # ID (number) of the pedestrian
        self.trajectory = [] # trajectory of the pedestrian