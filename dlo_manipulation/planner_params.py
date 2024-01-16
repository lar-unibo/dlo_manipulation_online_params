import numpy as np
from scipy.interpolate import splprep, splev

from dlo_manipulation.action_params import ActionFinderParamsGradient


class PlannerParamsRandomBending:
    def __init__(self, min_range=0.02, max_range=0.05, dtheta_range=np.pi / 3):
        self.min_range = min_range
        self.max_range = max_range
        self.dtheta_range = dtheta_range

    def random_index(self, dlo_0):
        num_nodes = dlo_0.shape[0]
        return int(np.random.randint(2, num_nodes - 3))

    def compute_disp(self, idx, dlo_0):
        range_motion = np.random.uniform(self.min_range, self.max_range)
        pick_dir = dlo_0[idx + 1] - dlo_0[idx]
        pick_dir = pick_dir / np.linalg.norm(pick_dir)
        return np.array([-pick_dir[1], pick_dir[0]]) * np.array([range_motion, range_motion])

    def random_dtheta(self):
        return np.random.uniform(-self.dtheta_range, self.dtheta_range)

    def disp_with_rotation(self, dlo_0):
        idx = self.random_index(dlo_0)
        disp = self.compute_disp(idx, dlo_0)
        dtheta = self.random_dtheta()
        return np.array([idx, disp[0], disp[1], dtheta])

    def disp_only(self, dlo_0):
        idx = self.random_index(dlo_0)
        disp = self.compute_disp(idx, dlo_0)
        return np.array([idx, disp[0], disp[1], 0.0])


class PlannerParamsBest:
    def __init__(
        self, checkpoint_path, optimization_steps=50, learning_rate=0.001, device="cpu", type_action="max_grad_norm"
    ):
        self.device = device

        self.af = ActionFinderParamsGradient(
            checkpoint_path=checkpoint_path,
            device=device,
            lr=learning_rate,
            num_steps=optimization_steps,
            verbose=True,
            type_action=type_action,
        )

        self.num_nodes = self.af.num_nodes

    def compute_length(self, shape):
        return np.sum(np.linalg.norm(np.diff(shape, axis=0), axis=1))

    def compute_spline(self, points, num_points=None, target_length=None):
        tck, u = splprep(np.array(points).T, u=None, k=3, s=0)
        if target_length is None:
            u_new = np.linspace(u.min(), u.max(), num_points)
        else:
            curr_length = self.compute_length(points)
            gap = (1 - target_length / curr_length) / 2
            u_new = np.linspace(u.min() + gap, u.max() - gap, num_points)

        return np.array(splev(u_new, tck, der=0)).T

    def exec_batch(self, dlo_0, kd=10, kb=0.5, m=0.05):
        # parameters
        params = np.array([kd, kb, m])

        #######################
        out_log = self.af.run_batch(dlo_0, params, indices=np.arange(self.num_nodes - 1))

        pred_list = [out_log[key]["pred"] for key in out_log.keys()]
        best_action_list = [out_log[key]["best_action"] for key in out_log.keys()]
        loss_list = [out_log[key]["loss_pred"] for key in out_log.keys()]

        best_loss_idx = np.argmin(loss_list)
        print(f"best_loss: idx {best_loss_idx}, value {loss_list[best_loss_idx]}")

        pred = pred_list[best_loss_idx]
        best_action = best_action_list[best_loss_idx]

        return pred, best_action, out_log

    ############################################################################################
    ######################################### PLOT #############################################
    def plot_batch_15(self, log_dict, save_path=None):
        self.af.plot_batch_15(log_dict, save_path=save_path)
