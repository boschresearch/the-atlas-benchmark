from collections import defaultdict, namedtuple
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])
MapRow = namedtuple('Row', ['id', 'origin_x', 'origin_y', 'theta', 'h', 'w', 'resolution'])
SurfClassRow = namedtuple('Row', ['id', 'color', 'name'])
SurfaceRow = namedtuple('Row', ['x', 'y', 'surf_class'])


class Reader(object):
    """Read trajnet files.

    :param scene_type: None -> numpy.array, 'rows' -> TrackRow and SceneRow, 'paths': grouped rows (primary pedestrian first)
    :param image_file: Associated image file of the scene
    """
    def __init__(self, input_file, scene_type=None, image_file=None):
        if scene_type is not None and scene_type not in {'rows', 'paths'}:
            raise Exception('scene_type not supported')
        self.scene_type = scene_type
        self.obstacle_id = None
        self.tracks_by_frame = defaultdict(list)
        self.tracks_by_person = defaultdict(list)
        self.scenes_by_id = dict()
        self.surfaces_by_id = defaultdict(list)
        self.surf_classes_by_id = defaultdict(list)
        self.map_data = None
        self.read_file(input_file)


    def read_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:
                    row = TrackRow(track['f'], track['p'], track['x'], track['y'],
                                   track.get('prediction_number'), track.get('scene_id'))
                    self.tracks_by_frame[row.frame].append(row)
                    self.tracks_by_person[row.pedestrian].append(row)
                    continue

                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'],
                                   scene['fps'], scene['tag'])
                    self.scenes_by_id[row.scene] = row

                surface = line.get('surface')
                if surface is not None:
                    row = SurfaceRow(surface['x'], surface['y'], surface['surf_class'])
                    self.surfaces_by_id[row.surf_class].append(row)

                surf_class = line.get('surf_class')
                if surf_class is not None:
                    row = SurfClassRow(surf_class['id'], surf_class['color'], surf_class['name'])
                    # new code!
                    if surf_class['name'] == 'obstacle':
                        self.obstacle_id = surf_class['id']
                    # new code end!
                    self.surf_classes_by_id[row.id].append(row)

                map = line.get('map')
                if map is not None:
                    row = MapRow(map['id'], map['origin_x'], map['origin_y'], map['theta'], map['h'],
                                 map['w'], map['resolution'])
                    self.map_data = row

    def surface_rows_to_map(self):
        semantic_classes = []
        f_s = np.zeros([len(self.surfaces_by_id), self.map_data.h, self.map_data.w])
        for s in self.surfaces_by_id:
            semantic_classes.append(s)
            for p in self.surfaces_by_id[s]:
                f_s[len(semantic_classes) - 1, p.x, p.y] = 1
        return f_s, semantic_classes

    def load_semantic_classes(self):
        semantic_classes = []
        colors = []
        for s in self.surf_classes_by_id:
            semantic_classes.append(s)
            colors.append(self.surf_classes_by_id[s][0].color)
        return semantic_classes, colors

    def track_rows_to_traj(self):
        paths = []
        for p in self.tracks_by_person:
            path = []
            for f in self.tracks_by_person[p]:
                path.append([f.x, f.y])
            paths.append(np.array(path))

        return paths

def read_map_from_json(path,sem_class):
    # sem_classes_input = Reader('thor_data/thor_sem_classes.ndjson')
    # all_semantic_classes, all_colors = sem_classes_input.load_semantic_classes()
    # print('Library of semantic classes includes: ', all_semantic_classes)

    # For each scene we read a feature map of this scene:
    # size num_sem_classes * Height * Width
    # Layer f_s[i] correspond to semantic class with ID contained in semantic_classes[i]

    # a = path.split('/')[-1]
    # b = int(a.split('_')[2])
    # if b != 3:        # thor experiment 1 and 2 share one map

    # else:
    f_s_input = Reader(path)
    sem_classes_input = Reader(sem_class)
    obstacle_id = sem_classes_input.obstacle_id
    f_s, semantic_classes = f_s_input.surface_rows_to_map()
    obstacle = f_s[0]
    p_ox = []
    p_oy = []
    for i in range(np.size(obstacle, axis=0)):
        for j in range(np.size(obstacle, axis=1)):
            if obstacle[i][j] == obstacle_id:     # obstacles are represented by 1, free space 0
                p_x = np.round((i * f_s_input.map_data.resolution + f_s_input.map_data.origin_x), decimals=2)
                p_y = np.round((j * f_s_input.map_data.resolution + f_s_input.map_data.origin_y), decimals=2)
                p_ox.append(p_x)
                p_oy.append(p_y)
    p_o = np.array([p_ox, p_oy])
    # plt.scatter(p_o[0],p_o[1])
    # plt.show()
    # world size grid map
    return p_o, obstacle

# This function is not used anywhere
def test_read_map():
    # Reading all existing semantic classes in the dataset, indexed with a unique ID and associated with a color
    sem_classes_input = Reader('json_data/thor_sem_classes.ndjson')
    all_semantic_classes, all_colors = sem_classes_input.load_semantic_classes()
    print('Library of semantic classes includes: ', all_semantic_classes)

    # For each scene we read a feature map of this scene:
    # size num_sem_classes * Height * Width
    # Layer f_s[i] correspond to semantic class with ID contained in semantic_classes[i]

    f_s_input = Reader('json_data/thor_Ex_1-f_s.ndjson')
    f_s, semantic_classes = f_s_input.surface_rows_to_map()
    colors = [all_colors[i] for i in semantic_classes]
    print('Loaded a semantic map of size', f_s.shape)
    print('This map has following semantic classes:', semantic_classes)
    print('Marked by the following colors:', colors)

    # Reading trajectories
    traj_input = Reader('json_data/thor_Ex_1_run_1-traj.ndjson')
    traj = traj_input.track_rows_to_traj()

    # Plotting semantic map and trajectories
    D_cum = np.zeros(f_s.shape[1:])
    for t in traj:
        for p in t:
            # convert (x,y) positions to (i,j) map coordinates
            i = round((p[0] - f_s_input.map_data.origin_x) / f_s_input.map_data.resolution)
            j = round((p[1] - f_s_input.map_data.origin_y) / f_s_input.map_data.resolution)
            D_cum[int(i), int(j)] += 1
    plt.imshow(np.log(1 + D_cum))
    string_name = 'results/thor_trajectories.png'
    plt.savefig(string_name, dpi=300)
    plot_sem_map(f_s, colors)
    string_name = 'results/thor_semantic_map.png'
    plt.savefig(string_name, dpi=300)
    plt.clf()
    return

# This function is not used anywhere
def plot_sem_map(f_s, colors):
    for f, c in zip(f_s, colors):
        x, y = np.nonzero(f)
        plt.scatter(y, x, color=c, s=0.01)

#
#
# The following functions were originally contained in the load_sample_json.py
#
#

# read the stream of detections from a json file
def read_json(input_file):
    traj = []
    with open(input_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            track = line.get('track')
            if track is not None:
                row = track['f'], track['p'], track['x'], track['y']
                traj.append(np.array(row))
    return np.array(traj)


# load map, transfer from pixel to world coordinate
def trans_map(path, h_matrix=None):
    I = mpimg.imread(path)  # size(I) is (480,640)
    if h_matrix:
        H = np.loadtxt(h_matrix)
    else:
        H = np.array([[1,0,0],[0,1,0],[0,0,1]])
    b = np.where(I == 1)  # gives the index of obstacle
    z = np.ones(np.size(b, axis=1))  # add z axis to pixel coordinate
    pixel = np.vstack((b, z))
    world_co = np.dot(H, pixel)
    world_re = world_co / world_co[2, :]
    world_d2 = world_re[0:2, :]
    return world_d2

def read_point_obstacles(path):
    obstacles = np.loadtxt(path)
    if len(np.shape(obstacles)) == 0:
        return []
    if len(np.shape(obstacles)) == 1:
        return [[ [obstacles[0],obstacles[1]], obstacles[2]]]
    # else, if there is more than one obstacle
    world_d2 = []
    for o in obstacles:
        world_d2.append([ [o[0],o[1]], o[2]])
    return world_d2

# Does not seem to be used anywhere
def world2pixel(world_d2, h_matrix):        # world_d2 should be (2, n)
    z = np.ones(np.size(world_d2, axis=1))
    world_extend = np.vstack((world_d2, z))
    H = np.loadtxt(h_matrix)
    H_inverse = np.linalg.inv(H)
    pixel = np.dot(H_inverse, world_extend)
    pixel_re = pixel / pixel[2, :]
    pixel = pixel_re[0:2, :]
    round_pixel = np.rint(pixel)
    picture = np.zeros((480,640))
    for i,j in zip(round_pixel[0], round_pixel[1]):
        picture[int(i)][int(j)] = 1
    return picture