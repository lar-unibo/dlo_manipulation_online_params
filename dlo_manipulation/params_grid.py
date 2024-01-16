import torch
import numpy as np
from dlo_manipulation.model import FCMul
from dlo_manipulation.dataset import DloSample
import matplotlib.pyplot as plt


class ParametersFinderGridSearch:
    def __init__(self, checkpoint_path, device="cpu", grid_size=10):
        self.device = device
        self.grid_size = grid_size

        # LOAD CHECKPOINT
        state = torch.load(checkpoint_path, map_location=self.device)
        self.num_nodes = state["num_points"]
        self.params_range = state["parameter_ranges"]

        # MODEL
        self.model = FCMul(n_pts=state["num_points"], pts_dim=state["dim_points"], hidden_dim=state["hidden_dim"])
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()

        # LOSS
        self.loss_fcn = lambda x, y: torch.mean(torch.linalg.norm(x - y, axis=-1))

        # SAMPLE OBJ
        self.sample_obj = DloSample(self.params_range)

        ###############################
        # GRID SEARCH
        kd_normalized_range = [0.0, 1.0]
        kb_normalized_range = [0.0, 1.0]
        self.dkd = np.linspace(kd_normalized_range[0], kd_normalized_range[1], self.grid_size)
        self.dkb = np.linspace(kb_normalized_range[0], kb_normalized_range[1], self.grid_size)

    def load_sample(self, file_path):
        dlo_0_n, dlo_1_n, action_n, _ = self.sample_obj.load_and_normalize_sample(file_path)

        if np.linalg.norm(dlo_0_n[0, :] - dlo_1_n[0, :]) > np.linalg.norm(dlo_0_n[0, :] - dlo_1_n[-1, :]):
            dlo_1_n = np.flip(dlo_1_n, axis=0)

        if not isinstance(dlo_0_n, torch.Tensor):
            dlo_0 = torch.from_numpy(dlo_0_n.copy()).float()
            dlo_0.unsqueeze_(0)
        if not isinstance(dlo_1_n, torch.Tensor):
            dlo_1 = torch.from_numpy(dlo_1_n.copy()).float()
            dlo_1.unsqueeze_(0)
        if not isinstance(action_n, torch.Tensor):
            action = torch.from_numpy(action_n.copy()).float()
            action.unsqueeze_(0)

        return dlo_0, dlo_1, action

    def plot_result(self, log, save_path=None):
        kd_param_range = self.params_range["kd"]
        kb_param_range = self.params_range["kb"]

        grid_img = log["err_img"]
        dlo_0 = log["dlo_0"]
        dlo_1 = log["dlo_1"]
        model_pred = log["model_pred"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(model_pred[:, 0], model_pred[:, 1], "o-", label="predicted (best)")
        ax1.plot(dlo_1[:, 0], dlo_1[:, 1], "o-", label="desired")
        ax1.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="init")
        ax1.axis("equal")
        ax1.legend(ncol=2)

        im = ax2.imshow(grid_img, extent=[0, 1, 0, 1], origin="lower")
        ax2.scatter(log["best_i_kb"] / self.grid_size, log["best_i_kd"] / self.grid_size, marker="x", color="red")
        plt.colorbar(im, ax=ax2)
        ax2.set_xlabel("KB")
        ax2.set_ylabel("KD")

        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        kd_labels = ["{:.2f}".format(x) for x in np.linspace(kd_param_range["min"], kd_param_range["max"], len(ticks))]
        kb_labels = ["{:.2f}".format(x) for x in np.linspace(kb_param_range["min"], kb_param_range["max"], len(ticks))]
        ax2.set_xticks(ticks)
        ax2.set_yticks(ticks)
        ax2.set_xticklabels(kb_labels)
        ax2.set_yticklabels(kd_labels)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close("all")
        else:
            plt.show()

    def plot_result_batch(self, log, save_path=None, kd_gt=None, kb_gt=None):
        kd_param_range = self.params_range["kd"]
        kb_param_range = self.params_range["kb"]
        grid_img = log["err_img"]
        fig = plt.figure()
        im = plt.imshow(grid_img, extent=[0, 1, 0, 1], origin="lower")
        plt.scatter(log["best_i_kb"] / self.grid_size, log["best_i_kd"] / self.grid_size, marker="x", color="red")

        if kd_gt is not None and kb_gt is not None:
            kd_i_gt = (kd_gt - kd_param_range["min"]) / (kd_param_range["max"] - kd_param_range["min"])
            kb_i_gt = (kb_gt - kb_param_range["min"]) / (kb_param_range["max"] - kb_param_range["min"])
            plt.scatter(kb_i_gt, kd_i_gt, marker="o", color="green")

        plt.colorbar(im)
        plt.xlabel("KB")
        plt.ylabel("KD")

        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        kd_labels = ["{:.2f}".format(x) for x in np.linspace(kd_param_range["min"], kd_param_range["max"], len(ticks))]
        kb_labels = ["{:.2f}".format(x) for x in np.linspace(kb_param_range["min"], kb_param_range["max"], len(ticks))]
        ax = plt.gca()
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(kb_labels)
        ax.set_yticklabels(kd_labels)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close("all")

    def batch_gridsearch_fixed_mass(self, data_files, mass):
        grid_img = np.zeros((self.grid_size, self.grid_size))
        for i_kd, kd in enumerate(self.dkd):
            for i_kb, kb in enumerate(self.dkb):
                batch_loss = 0.0
                for i, data in enumerate(data_files):
                    dlo_0, dlo_1, action = data

                    params = torch.tensor([kd, kb, mass]).float().unsqueeze(0)

                    pred = self.model(dlo_0, action, params)

                    l = self.loss_fcn(pred, dlo_1)
                    batch_loss += l.item()

                grid_img[i_kd, i_kb] = batch_loss / len(data_files)  # KD rows, KB columns

        # sort list config_err by err key
        best_i_kd, best_i_kb = np.unravel_index(grid_img.argmin(), grid_img.shape)

        best_kd_denormalized = self.sample_obj.denormalize_kd(self.dkd[best_i_kd])
        best_kb_denormalized = self.sample_obj.denormalize_kb(self.dkb[best_i_kb])

        log_output = {
            "num_nodes": self.num_nodes,
            "mass": mass,
            "kd_range": self.dkd,
            "kb_range": self.dkb,
            "best_kd": self.dkd[best_i_kd],
            "best_i_kd": best_i_kd,
            "best_kb": self.dkb[best_i_kb],
            "best_i_kb": best_i_kb,
            "best_err": grid_img[best_i_kd, best_i_kb],
            "err_img": grid_img,
        }

        return (best_kd_denormalized, best_kb_denormalized), log_output

    def single_gridsearch_fixed_mass(self, dlo_0, dlo_1, action, mass):
        grid_img = np.zeros((self.grid_size, self.grid_size))
        grid_pred = np.zeros((self.grid_size, self.grid_size, self.num_nodes, 2))
        for i_kd, kd in enumerate(self.dkd):
            for i_kb, kb in enumerate(self.dkb):
                params = torch.tensor([kd, kb, mass]).float().unsqueeze(0)

                pred = self.model(dlo_0, action, params)
                l = self.loss_fcn(pred, dlo_1)

                grid_img[i_kd, i_kb] = l.detach().numpy()  # KD rows, KB columns
                grid_pred[i_kd, i_kb] = pred.squeeze().detach().numpy()

        # sort list config_err by err key
        best_i_kd, best_i_kb = np.unravel_index(grid_img.argmin(), grid_img.shape)

        best_kd_denormalized = self.sample_obj.denormalize_kd(self.dkd[best_i_kd])
        best_kb_denormalized = self.sample_obj.denormalize_kb(self.dkb[best_i_kb])

        log_output = {
            "num_nodes": self.num_nodes,
            "mass": mass,
            "kd_range": self.dkd,
            "kb_range": self.dkb,
            "best_kd": self.dkd[best_i_kd],
            "best_i_kd": best_i_kd,
            "best_kb": self.dkb[best_i_kb],
            "best_i_kb": best_i_kb,
            "best_err": grid_img[best_i_kd, best_i_kb],
            "err_img": grid_img,
            "grid_pred": grid_pred,
            "dlo_0": dlo_0.squeeze().detach().cpu().numpy(),
            "dlo_1": dlo_1.squeeze().detach().cpu().numpy(),
            "action": action,
            "model_pred": grid_pred[best_i_kd, best_i_kb],
        }

        return (best_kd_denormalized, best_kb_denormalized), log_output
