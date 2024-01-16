import os, pickle, glob
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DloParams:
    def __init__(self, parameter_ranges=None):
        def read_parameter_ranges(param):
            return parameter_ranges[param]["min"], parameter_ranges[param]["max"] - parameter_ranges[param]["min"]

        self.kb_min, self.kb_range = read_parameter_ranges("kb")
        self.kd_min, self.kd_range = read_parameter_ranges("kd")
        self.m_min, self.m_range = read_parameter_ranges("m")

    def normalize(self, p):
        p1 = self.normalize_damping(p[0])
        p2 = self.normalize_bending(p[1])
        p3 = self.normalize_mass(p[2])
        return np.stack([p1, p2, p3], axis=-1)

    def normalize_damping(self, p):
        return (p - self.kd_min) / self.kd_range

    def normalize_bending(self, p):
        return (p - self.kb_min) / self.kb_range

    def normalize_mass(self, p):
        return (p - self.m_min) / self.m_range

    def denormalize(self, p):
        p1 = self.denormalize_damping(p[0])
        p2 = self.denormalize_bending(p[1])
        p3 = self.denormalize_mass(p[2])
        return np.stack([p1, p2, p3], axis=-1)

    def denormalize_damping(self, p):
        return p * self.kd_range + self.kd_min

    def denormalize_bending(self, p):
        return p * self.kb_range + self.kb_min

    def denormalize_mass(self, p):
        return p * self.m_range + self.m_min


class DloAction:
    def __init__(self, num_points=16, scale_action=False):
        self.num_points = num_points
        self.scale_action = scale_action
        self.rot_check_flag = False
        self.cs0 = None
        self.csR = None
        self.angle_scale = np.pi / 4.0
        self.disp_scale = 0.1

    def set_normalize_factor(self, csR, cs0):
        self.csR = csR
        self.cs0 = cs0

    def set_rot_flag(self, rot_flag):
        self.rot_check_flag = rot_flag

    # ACTION
    def normalize(self, dlo_0, a):
        a1 = self.normalize_action_idx(a[0])

        # compute the target position of the active edge based on the initial shape and the action
        points_grasp, points_place = self.compute_edges_points_from_action(dlo_0, a)

        points_grasp_up = np.dot(self.csR, (points_grasp - self.cs0).T).T
        points_place_up = np.dot(self.csR, (points_place - self.cs0).T).T

        action_new = self.compute_action_from_edges_points(points_grasp_up, points_place_up)

        # scale values if necessary
        a23 = self.scale_disp(action_new[:2])
        a4 = self.scale_angle(action_new[-1])

        return np.stack([a1, a23[0], a23[1], a4], axis=-1)

    def normalize_action_idx(self, idx):
        if self.rot_check_flag is None:
            raise ValueError("rot_check_flag is None")

        if self.rot_check_flag:
            idx = (self.num_points - 2) - idx  # reverse the index of the active edge

        return idx / (self.num_points - 1.0)

    def scale_disp(self, disp):
        if self.scale_action:
            disp /= np.array([self.disp_scale, self.disp_scale])

        return disp

    def scale_angle(self, theta):
        if self.scale_action:
            theta /= self.angle_scale
        return theta

    # ACTION
    def denormalize(self, dlo_0_n, a):
        a1 = self.denormalize_action_idx(a[0])

        a23 = a[1:3]
        a4 = a[3]
        if self.scale_action:
            a23 = a23 * np.array([self.disp_scale, self.disp_scale])
            a4 = a4 * self.angle_scale

        a_new = np.array([a1, a23[0], a23[1], a4])

        # compute the target position of the active edge based on the initial shape and the action
        points_grasp, points_place = self.compute_edges_points_from_action(dlo_0_n, a_new)

        points_grasp_up = np.dot(self.csR.T, points_grasp.T).T + self.cs0
        points_place_up = np.dot(self.csR.T, points_place.T).T + self.cs0

        action_new = self.compute_action_from_edges_points(points_grasp_up, points_place_up)

        return np.stack([a1, action_new[0], action_new[1], action_new[2]], axis=-1)

    def denormalize_action_idx(self, idx):
        idx = idx * (self.num_points - 1.0)

        if self.rot_check_flag:
            idx = (self.num_points - 2) - idx  # reverse the index of the active edge

        return idx

    def compute_edges_points_from_action(self, init_pos, action):
        idx = int(action[0])
        dtheta = action[3]

        node_0_pos = init_pos[idx, :]
        node_1_pos = init_pos[idx + 1, :]

        edge_pos = (node_1_pos + node_0_pos) / 2
        edge_dir = node_1_pos - node_0_pos
        edge_len = np.linalg.norm(edge_dir)
        edge_dir = edge_dir / edge_len

        # new pos
        new_edge_pos_x = edge_pos[0] + action[1]
        new_edge_pos_y = edge_pos[1] + action[2]
        new_edge_pos = np.array([new_edge_pos_x, new_edge_pos_y])

        # new dir
        new_edge_x = edge_dir[0] * np.cos(dtheta) - edge_dir[1] * np.sin(dtheta)
        new_edge_y = edge_dir[0] * np.sin(dtheta) + edge_dir[1] * np.cos(dtheta)
        new_edge_dir = np.array([new_edge_x, new_edge_y])

        pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2
        pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2

        points_grasp = np.array([node_0_pos, node_1_pos]).squeeze()
        points_place = np.array([pos0_tgt, pos1_tgt]).squeeze()
        return points_grasp, points_place

    def compute_action_from_edges_points(self, points_grasp, points_place):
        center_grasp_up = points_grasp.mean(axis=0, keepdims=True)
        center_place_up = points_place.mean(axis=0, keepdims=True)

        dir_grasp_up = points_grasp[1, :] - points_grasp[0, :]
        dir_grasp_up = dir_grasp_up / np.linalg.norm(dir_grasp_up)

        dir_place_up = points_place[1, :] - points_place[0, :]
        dir_place_up = dir_place_up / np.linalg.norm(dir_place_up)

        angle = np.arctan2(dir_place_up[1], dir_place_up[0]) - np.arctan2(dir_grasp_up[1], dir_grasp_up[0])
        disp_x = center_place_up[0, 0] - center_grasp_up[0, 0]
        disp_y = center_place_up[0, 1] - center_grasp_up[0, 1]
        return np.array([disp_x, disp_y, angle])


class DloSample:
    def __init__(self, parameter_ranges, num_points=16, scale_action=True):
        self.parameter_ranges = parameter_ranges
        if self.parameter_ranges is not None:
            self.dlo_params = DloParams(parameter_ranges)

        self.dlo_action = DloAction(num_points=num_points, scale_action=scale_action)

        self.num_points = num_points

        self.cs0 = None
        self.csR = None
        self.rot_check_flag = None
        # self.yaxis_flipped = None

    def compute_edge_target_position(self, init_pos, edge_idx, dx, dy, dtheta):
        node_0 = edge_idx
        node_1 = edge_idx + 1

        node_0_pos = init_pos[node_0, :]
        node_1_pos = init_pos[node_1, :]

        edge_pos = (node_1_pos + node_0_pos) / 2
        edge_dir = node_1_pos - node_0_pos
        edge_len = np.linalg.norm(edge_dir)
        edge_dir = edge_dir / edge_len

        # new pos
        new_edge_pos_x = edge_pos[0] + dx
        new_edge_pos_y = edge_pos[1] + dy
        new_edge_pos = np.array([new_edge_pos_x, new_edge_pos_y, 0.0])

        # new dir
        new_edge_x = edge_dir[0] * np.cos(dtheta) - edge_dir[1] * np.sin(dtheta)
        new_edge_y = edge_dir[0] * np.sin(dtheta) + edge_dir[1] * np.cos(dtheta)
        new_edge_dir = np.array([new_edge_x, new_edge_y, 0.0])

        # new pos node 0
        pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2

        # new pos node 1
        pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2

        return pos0_tgt, pos1_tgt

    def load_sample(self, file):
        data = pickle.load(open(file, "rb"))

        dlo_0 = data["sys_before"][:, :2]
        dlo_1 = data["sys_after"][:, :2]

        if np.linalg.norm(dlo_0[0, :] - dlo_1[0, :]) > np.linalg.norm(dlo_0[0, :] - dlo_1[-1, :]):
            dlo_1 = np.flip(dlo_1, axis=0)

        if "action" in data:
            action = data["action"]
            idx = action["idx"]
            action_dx = action["action_dx"]
            action_dy = action["action_dy"]
            action_dtheta = action["action_dtheta"]
            action = [idx, action_dx, action_dy, action_dtheta]
        elif "action_dx" in data:
            action = [data["idx"], data["action_dx"], data["action_dy"], data["action_dtheta"]]
        else:
            action = [data["idx"], data["dx"], data["dy"], data["dtheta"]]

        if "params" in data:
            params = [data["params"]["kd"], data["params"]["kb"], data["params"]["m"]]
        else:
            params = [-1.0, -1.0, -1.0]

        return np.array(dlo_0), np.array(dlo_1), np.array(action), np.array(params)

    def load_and_normalize_sample(self, path):
        dlo_0, dlo_1, action, param = self.load_sample(path)
        dlo_0_n, dlo_1_n, action_n, param_n = self.normalize(dlo_0, dlo_1, action, param)
        return dlo_0_n, dlo_1_n, action_n, param_n

    def normalize(self, dlo_0, dlo_1, action, param):
        # compute normalization factors
        self.cs0, self.csR = self.compute_normalize_factor(dlo_0)
        self.dlo_action.set_normalize_factor(self.csR, self.cs0)

        # dlo shape
        dlo_0_n = self.normalize_dlo(dlo_0)
        dlo_1_n = self.normalize_dlo(dlo_1)

        # check rot
        dlo_0_n, dlo_1_n = self.check_rot_and_flip(dlo_0_n, dlo_1_n)
        self.dlo_action.set_rot_flag(self.rot_check_flag)

        # action
        # action = [14, -0.02, -0.02, 0.0]
        action_n = self.dlo_action.normalize(dlo_0, action)

        # params
        if self.parameter_ranges is None:
            param_n = np.array([-1.0, -1.0, -1.0])
        else:
            param_n = self.dlo_params.normalize(param)

        return dlo_0_n, dlo_1_n, action_n, param_n

    def compute_normalize_factor(self, dlo_0):
        cs0 = np.mean(dlo_0, axis=0, keepdims=True)

        dlo_0_centred = dlo_0 - cs0
        cov = dlo_0_centred.T @ dlo_0_centred
        eigval, eigvec = np.linalg.eig(cov)
        csR = eigvec[:, np.argsort(eigval)[::-1]].T

        return cs0, csR

    def normalize_dlo(self, dlo):
        if self.cs0 is None or self.csR is None:
            raise ValueError("cs0 or csR is None")

        return np.dot(self.csR, (dlo - self.cs0).T).T

    def normalize_sample_action(self, dlo_0, action):
        return self.dlo_action.normalize(dlo_0, action)

    def denormalize_sample_action(self, dlo_0_n, action_n):
        return self.dlo_action.denormalize(dlo_0_n, action_n)

    def check_rot_and_flip(self, dlo_0_n, dlo_1_n):
        # flipping the dlo of the wrong order and adjusting the action correspondingly
        self.rot_check_flag = dlo_0_n[0, 0] > 0.0
        if self.rot_check_flag:
            return dlo_0_n[::-1], dlo_1_n[::-1]

        return dlo_0_n, dlo_1_n

    """
    def conditionally_flip_yaxis_dlo(self, dlo_0_n, dlo_1_n):
        # flip y-axis such that action can only point upwards
        if self.yaxis_flipped is None:
            raise ValueError("yaxis_flipped is None")

        if self.yaxis_flipped:
            dlo_0_n[:, 1] = -dlo_0_n[:, 1]
            dlo_1_n[:, 1] = -dlo_1_n[:, 1]
        return dlo_0_n, dlo_1_n
    """

    ########################################################
    # DENORMALIZE

    def denormalize_dlo(self, dlo):
        # if self.yaxis_flipped:
        #    dlo[:, 1] = -dlo[:, 1]

        if self.rot_check_flag:
            dlo = dlo[::-1]

        return (self.csR.T @ dlo.T).T + self.cs0

    def denormalize(self, dlo_0_n, dlo_1_n, action_n, params_n):
        dlo_0 = self.denormalize_dlo(dlo_0_n)
        dlo_1 = self.denormalize_dlo(dlo_1_n)
        action = self.dlo_action.denormalize(dlo_0_n, action_n)

        if self.parameter_ranges is None:
            params = np.array([-1.0, -1.0, -1.0])
        else:
            params = self.dlo_params.denormalize(params_n)

        return dlo_0, dlo_1, action, params


class DloDataset(Dataset, DloSample):
    def __init__(self, dataset_path, parameter_ranges, num_points=16, augment=False, scale_action=True):
        super().__init__(parameter_ranges, num_points=num_points, scale_action=scale_action)
        self.num_points = num_points
        self.augment = augment

        data_files = glob.glob(os.path.join(dataset_path, "*.pickle"))
        self.data_samples = self.preprocess(data_files)

    def process_sample(self, file):
        dlo_0, dlo_1, action, params = self.load_sample(file)
        dlo_0_n, dlo_1_n, action_n, params_n = self.normalize(dlo_0, dlo_1, action, params)

        # tensors
        dlo_0_n = torch.from_numpy(dlo_0_n.copy()).float()
        dlo_1_n = torch.from_numpy(dlo_1_n.copy()).float()
        action_n = torch.from_numpy(action_n.copy()).float()
        params_n = torch.from_numpy(params_n.copy()).float()

        return dlo_0_n, dlo_1_n, action_n, params_n

    def preprocess(self, data_files):
        data_samples = []
        for it, file in enumerate(data_files):
            # try:
            dlo_0_n, dlo_1_n, action_n, params_n = self.process_sample(file)
            """
            except Exception as e:
                print("Error in processing file: {}".format(file))
                print(e)
                continue
            """
            data_samples.append([dlo_0_n, dlo_1_n, action_n, params_n])

        if self.augment:
            aug = [[d[0], d[0], d[2] * np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), d[3]] for d in data_samples]
            data_samples += aug
        return data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]
