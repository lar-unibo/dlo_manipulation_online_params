import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

from dlo_manipulation.action import ActionFinderGradient


class Planner:
    def __init__(
        self,
        checkpoint_path,
        optimization_steps=50,
        learning_rate=0.001,
        device="cpu",
    ):
        self.device = device

        self.af = ActionFinderGradient(
            checkpoint_path=checkpoint_path,
            device=device,
            lr=learning_rate,
            num_steps=optimization_steps,
            verbose=True,
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

    def check_smoothness(self, dlo):
        def cosine_sim(a, b):
            return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

        dirs = np.diff(dlo, axis=0)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        for it in range(1, len(dirs) - 1):
            prev = dirs[it - 1]
            curr = dirs[it]
            next = dirs[it + 1]
            s = cosine_sim(prev, curr) * cosine_sim(curr, next)
            if s < 0:
                return False

        return True

    def get_init_and_target(self, dlo_0, dlo_1):
        dlo_0 = self.compute_spline(dlo_0, num_points=self.num_nodes)
        dlo_1 = self.compute_spline(dlo_1, num_points=self.num_nodes, target_length=self.compute_length(dlo_0))

        if np.linalg.norm(dlo_1 - dlo_0) > np.linalg.norm(np.flip(dlo_1, axis=0) - dlo_0):
            dlo_1 = np.flip(dlo_1, axis=0)

        return dlo_0, dlo_1

    def exec_single(self, dlo_0, dlo_1, idx=0, kd=10, kb=0.5, m=0.05, plot_loss=False):
        # set init and target
        dlo_0, dlo_1 = self.get_init_and_target(dlo_0, dlo_1)

        # parameters to use
        params = np.array([kd, kb, m])

        #######################
        out_log = self.af.run(dlo_0, dlo_1, params, idx=idx)

        pred = out_log["pred"]
        best_action = out_log["best_action"]
        opt_log = out_log["opt_log"]

        if plot_loss:
            losses = [opt_log[key]["loss"] for key in opt_log.keys()]
            Planner.plot_loss(losses)

        return pred, best_action, out_log

    def exec_loop_single(self, dlo_0, dlo_1, kd=10, kb=0.5, m=0.05):
        batch_size = self.num_nodes - 1
        out_dict = {}
        for i in range(batch_size):
            _, _, out_log_dict = self.exec_single(dlo_0, dlo_1, idx=i, kd=kd, kb=kb, m=m)
            out_dict[i] = out_log_dict

        losses = [out_dict[key]["loss_pred"] for key in out_dict.keys()]
        best_loss_idx = np.argmin(losses)
        print(f"best_loss: idx {best_loss_idx}, value {losses[best_loss_idx]}")

        pred_best = out_dict[best_loss_idx]["pred"]
        action_best = out_dict[best_loss_idx]["best_action"]

        return pred_best, action_best, out_dict

    def exec_batch(self, dlo_0, dlo_1, kd=10, kb=0.5, m=0.05):
        # set init and target
        dlo_0, dlo_1 = self.get_init_and_target(dlo_0, dlo_1)

        # parameters
        params = np.array([kd, kb, m])

        #######################
        out_log = self.af.run_batch(dlo_0, dlo_1, params, indices=np.arange(self.num_nodes - 1))

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

    def plot_single(self, log_dict):
        idx = log_dict["idx"]
        dlo_0 = log_dict["dlo_0"]
        dlo_1 = log_dict["dlo_1"]
        pred = log_dict["pred"]
        pred_init = log_dict["pred_init"]

        init_pos_idx = (dlo_0[idx] + dlo_0[idx + 1]) / 2
        target_pos_idx = (dlo_1[idx] + dlo_1[idx + 1]) / 2
        pred_pos_idx = (pred[idx] + pred[idx + 1]) / 2

        # plot
        plt.figure("plot single")
        plt.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", c="red", label="init")
        plt.plot(dlo_1[:, 0], dlo_1[:, 1], "o-", c="green", label="target")
        plt.plot(pred[:, 0], pred[:, 1], "o-", c="blue", label="pred")
        plt.plot(pred_init[:, 0], pred_init[:, 1], "o-", label="pred_init")

        plt.scatter(init_pos_idx[0], init_pos_idx[1], marker="X", s=200, label="init_pos")
        plt.scatter(target_pos_idx[0], target_pos_idx[1], marker="X", s=200, label="target_pos")
        plt.scatter(pred_pos_idx[0], pred_pos_idx[1], marker="X", s=200, label="pred_pos")

        plt.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_loss(loss_arr):
        fig = plt.figure("loss curve", figsize=(6, 6))
        plt.plot(np.arange(len(loss_arr)), loss_arr, ".-")
        plt.xlabel("epochs")
        plt.tight_layout()
        # plt.show()

    @staticmethod
    def rot(v, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return np.dot(R, v)

    @staticmethod
    def plot_batch_15(log_dict, save_path=None):
        def compute_edge_target_position(init_pos, edge_idx, dx, dy, dtheta):
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

        rows = 3
        cols = 5
        scatter_size = 20
        fig, axs = plt.subplots(rows, cols, figsize=(18, 9))

        for idx, v in log_dict.items():
            dlo_0 = v["dlo_0"]
            dlo_1 = v["dlo_1"]
            pred = v["pred"]
            pred_init = v["pred_init"]
            loss = v["loss_pred"]

            best_action = v["best_action"]
            best_dx = best_action[1]
            best_dy = best_action[2]
            best_dtheta = best_action[3]

            point_new1, point_new2 = compute_edge_target_position(dlo_0, idx, best_dx, best_dy, best_dtheta)

            row = idx // cols
            col = idx % cols

            axs[row, col].plot(dlo_0[:, 0], dlo_0[:, 1], ".-", c="red", label="init", linewidth=0.5)
            axs[row, col].plot(dlo_1[:, 0], dlo_1[:, 1], ".-", c="green", label="target", linewidth=0.5)
            axs[row, col].plot(pred[:, 0], pred[:, 1], ".-", c="blue", label="pred", linewidth=1.0, zorder=100)
            axs[row, col].plot(
                pred_init[:, 0], pred_init[:, 1], ".-", c="gray", label="pred_init", linewidth=0.5, alpha=0.5
            )

            axs[row, col].scatter(dlo_0[idx, 0], dlo_0[idx, 1], s=scatter_size, c="cyan", marker="X", zorder=100)
            axs[row, col].scatter(
                dlo_0[idx + 1, 0], dlo_0[idx + 1, 1], s=scatter_size, c="cyan", marker="X", zorder=100
            )

            axs[row, col].scatter(pred[idx, 0], pred[idx, 1], s=scatter_size, c="cyan", marker="X", zorder=100)
            axs[row, col].scatter(pred[idx + 1, 0], pred[idx + 1, 1], s=scatter_size, c="cyan", marker="X", zorder=100)
            axs[row, col].plot(
                [point_new1[0], point_new2[0]], [point_new1[1], point_new2[1]], c="magenta", linewidth=3, zorder=100
            )

            axs[row, col].set_title(f"idx {idx}, loss {loss:.3f}")
            axs[row, col].axis("on")
            axs[row, col].axis("equal")

            if idx == 0:
                axs[row, col].legend(ncol=2, fontsize=8)

        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches="tight")
