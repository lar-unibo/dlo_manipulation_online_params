import torch
import numpy as np
import matplotlib.pyplot as plt

from dlo_manipulation.model import FCMul
from dlo_manipulation.dataset import DloSample


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


class ActionFinderParamsGradient:
    def __init__(
        self, checkpoint_path, device="cpu", lr=1e-3, num_steps=500, verbose=False, type_action="max_bending"
    ):
        self.device = device
        self.lr = lr
        self.num_steps = num_steps
        self.verbose = verbose

        # STATE
        state = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.num_nodes = state["num_points"]

        # MODEL
        self.model = FCMul(n_pts=state["num_points"], pts_dim=state["dim_points"], hidden_dim=state["hidden_dim"])
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.sample_obj = DloSample(state["parameter_ranges"])

        print("*" * 50)
        print("Action Finder")
        print(f"num_steps: {self.num_steps}")
        print(f"lr: {self.lr}")
        print("*" * 50)

        if type_action == "max_bending":
            self.finder = ActionMaxBending(
                model=self.model, num_steps=self.num_steps, lr=self.lr, verbose=self.verbose, device=self.device
            )
        elif type_action == "max_grad_norm":
            self.finder = ActionMaxGradNorm(
                model=self.model, num_steps=self.num_steps, lr=self.lr, verbose=self.verbose
            )
        else:
            raise ValueError(f"Invalid type_action: {type_action}")

    def sample_init_action_given_idx(self, dlo_0, idx):
        np.random.seed()  # to reeseed allowing different initial displacements for every process

        # DISPLACEMENT
        dir = dlo_0[idx + 1] - dlo_0[idx]
        dir = dir / np.linalg.norm(dir)
        target_dir = np.array([dir[1], -dir[0]])  # 90deg rotation
        target_dir = target_dir / np.linalg.norm(target_dir)

        f = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.1)
        disp = f * target_dir

        # theta
        theta = 0.0

        return disp, theta

    def run_from_file(self, file_path, params_real):
        dlo_0, _, _, _ = self.sample_obj.load_sample(file_path)
        return self.run(dlo_0, params_real, idx=0)

    def run_batch_from_file(self, file_path, params_real):
        dlo_0, _, _, _ = self.sample_obj.load_sample(file_path)
        return self.run_batch(dlo_0, params_real, indices=np.arange(len(dlo_0) - 1))

    def run(self, dlo_0, params_real, idx):
        # init action
        disp, theta = self.sample_init_action_given_idx(dlo_0, idx)
        init_action = np.array([idx, disp[0], disp[1], theta])

        ############################
        # NORMALIZATION
        dlo_0_n, _, init_action_n, params_n = self.sample_obj.normalize(
            dlo_0, np.zeros(dlo_0.shape), init_action, params_real
        )

        # to tensor
        dlo_0_n = torch.from_numpy(dlo_0_n.copy()).float().unsqueeze_(0)
        params_n = torch.from_numpy(params_n.copy()).float().unsqueeze_(0)
        init_action_tn = torch.from_numpy(init_action_n.copy()).float().unsqueeze_(0)

        # ******************************************************************************************
        # find action
        opt_log = self.finder.find_action(dlo_0_n, init_action_tn, params_n)
        # find action
        # ******************************************************************************************

        # best action
        loss = np.array([opt_log[k]["losses"] for k in opt_log.keys()]).squeeze()
        best_action_tn = opt_log[np.argmin(loss)]["action"]

        # predictions
        pred_n = to_numpy(self.model(dlo_0_n, best_action_tn, params_n).squeeze())
        pred_init_n = to_numpy(self.model(dlo_0_n, init_action_tn, params_n).squeeze())

        # to numpy
        best_action_n = to_numpy(best_action_tn.squeeze())

        ############################
        best_action = self.sample_obj.denormalize_sample_action(best_action_n)
        pred = self.sample_obj.denormalize_dlo(pred_n)
        pred_init = self.sample_obj.denormalize_dlo(pred_init_n)

        # loss
        loss_action_init = self.finder.loss_fcn(torch.from_numpy(pred_init).unsqueeze(0)).item()
        loss_pred = self.finder.loss_fcn(torch.from_numpy(pred).unsqueeze(0)).item()

        print(f"loss_action_init: {loss_action_init}")
        print(f"loss_pred: {loss_pred}")

        output_log = {
            "dlo_0": dlo_0,
            "pred": pred,
            "pred_init": pred_init,
            "opt_log": opt_log,
            "best_action": best_action,
            "best_action_normalized": best_action_n,
            "init_action": init_action,
            "init_action_normalized": init_action_n,
            "loss_action_init": loss_action_init,
            "loss_pred": loss_pred,
        }

        return output_log

    def run_batch(self, dlo_0, params_real, indices):
        # init action and normalization
        init_action, init_action_n = [], []
        dlo_0_list, params_list = [], []
        for idx in indices:
            disp, theta = self.sample_init_action_given_idx(dlo_0, idx)
            act = np.array([idx, disp[0], disp[1], theta])

            init_action.append(act)

            dlo_0_n, _, act_n, params_n = self.sample_obj.normalize(dlo_0, np.zeros(dlo_0.shape), act, params_real)

            init_action_n.append(act_n)
            dlo_0_list.append(dlo_0_n)
            params_list.append(params_n)

        init_action = np.array(init_action)
        init_action_n = np.array(init_action_n)
        dlo_0_n = np.array(dlo_0_list)
        params_n = np.array(params_list)

        # to tensor
        dlo_0_n = torch.from_numpy(dlo_0_n.copy()).float()
        params_n = torch.from_numpy(params_n.copy()).float()
        init_action_tn = torch.from_numpy(init_action_n.copy()).float()

        # ******************************************************************************************

        # find action
        opt_log = self.finder.find_action(dlo_0_n, init_action_tn, params_n)

        # ******************************************************************************************

        losses = np.array([opt_log[k]["losses"] for k in opt_log.keys()]).squeeze()

        output_batch_log = {}
        for i in range(len(indices)):
            loss_i = losses[i, :]
            action_i = [opt_log[k]["action"][i] for k in opt_log.keys()]

            # best action
            best_loss_idx = np.argmin(loss_i)
            best_action_tn = action_i[best_loss_idx]

            # predictions
            pred_n = to_numpy(self.model(dlo_0_n[i], best_action_tn, params_n[i]).squeeze())
            pred_init_n = to_numpy(self.model(dlo_0_n[i], init_action_tn[i], params_n[i]).squeeze())

            # to numpy
            best_action_n = to_numpy(best_action_tn.squeeze())

            ############################
            best_action = self.sample_obj.denormalize_sample_action(best_action_n)
            pred = self.sample_obj.denormalize_dlo(pred_n)
            pred_init = self.sample_obj.denormalize_dlo(pred_init_n)

            # loss
            loss_action_init = torch.sum(self.finder.loss_fcn(torch.from_numpy(pred_init).unsqueeze(0))).item()
            loss_pred = torch.sum(self.finder.loss_fcn(torch.from_numpy(pred).unsqueeze(0))).item()

            output_batch_log[indices[i]] = {
                "dlo_0": dlo_0,
                "pred": pred,
                "pred_init": pred_init,
                "opt_log": {"loss": loss_i, "action": action_i},
                "best_action": best_action,
                "best_action_normalized": best_action_n,
                "init_action": init_action,
                "init_action_normalized": init_action_n,
                "loss_action_init": loss_action_init,
                "loss_pred": loss_pred,
            }

        return output_batch_log

    def plot_log(self, log_dict):
        opt_log = log_dict["opt_log"]
        pred = log_dict["pred"]
        pred_init = log_dict["pred_init"]
        dlo_0 = log_dict["dlo_0"]

        best_action = log_dict["best_action"]

        x_axis = np.array(list(opt_log.keys()))

        # LOG DICT
        actions = np.array([to_numpy(opt_log[k]["action"].squeeze()) for k in opt_log.keys()])
        loss_list = np.array([opt_log[k]["losses"] for k in opt_log.keys()]).squeeze()
        action_x_list = np.array([a[1] for a in actions])
        action_y_list = np.array([a[2] for a in actions])
        action_theta_list = np.array([a[3] for a in actions])

        # best loss
        best_loss_idx = np.argmin(loss_list)
        best_x = action_x_list[best_loss_idx]
        best_y = action_y_list[best_loss_idx]
        best_theta = action_theta_list[best_loss_idx]

        # EDGE ACTION
        idx = int(best_action[0])
        action_pick = np.array([dlo_0[idx], dlo_0[idx + 1]])
        action_p0, action_p1 = self.sample_obj.compute_edge_target_position(
            dlo_0, idx, best_action[1], best_action[2], best_action[3]
        )
        action_place = np.array([action_p0, action_p1])

        fig = plt.figure(figsize=(10, 6))
        plt.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="dlo_0")
        plt.plot(pred_init[:, 0], pred_init[:, 1], "o-", label="pred_init")
        plt.plot(pred[:, 0], pred[:, 1], "o-", label="pred")
        plt.plot(action_pick[:, 0], action_pick[:, 1], "o-", label="action_pick", linewidth=3, color="red")
        plt.plot(action_place[:, 0], action_place[:, 1], "o-", label="action_place", linewidth=3, color="green")
        plt.axis("equal")
        plt.legend()

        fig, axs = plt.subplots(1, 4, figsize=(15, 6))
        axs[0].plot(x_axis, action_x_list, label=f"trained ({best_x:.2f})")
        axs[0].legend()
        axs[0].set_title("disp_x")
        axs[0].set_ylim([-1.0, 1.0])

        axs[1].plot(x_axis, action_y_list, label=f"trained ({best_y:.2f})")
        axs[1].legend()
        axs[1].set_title("disp_y")
        axs[1].set_ylim([-1.0, 1.0])

        axs[2].plot(x_axis, action_theta_list, label=f"trained ({best_theta:.2f})")
        axs[2].legend()
        axs[2].set_title("theta")
        axs[2].set_ylim([0.0, 1.0])

        axs[3].plot(x_axis, loss_list)
        axs[3].set_title("loss")

        plt.tight_layout()
        plt.show()

    def plot_batch_15(self, log_dict, save_path=None):
        rows = 3
        cols = 5
        scatter_size = 20
        fig, axs = plt.subplots(rows, cols, figsize=(18, 9))

        for idx, v in log_dict.items():
            dlo_0 = v["dlo_0"]
            pred = v["pred"]
            pred_init = v["pred_init"]
            loss = v["loss_pred"]

            best_action = v["best_action"]

            # EDGE ACTION
            action_pick = np.array([dlo_0[idx], dlo_0[idx + 1]])
            action_p0, action_p1 = self.sample_obj.compute_edge_target_position(
                dlo_0, int(best_action[0]), best_action[1], best_action[2], best_action[3]
            )
            action_place = np.array([action_p0, action_p1])

            row = idx // cols
            col = idx % cols

            axs[row, col].plot(dlo_0[:, 0], dlo_0[:, 1], ".-", c="red", label="init", linewidth=0.5)
            axs[row, col].plot(pred[:, 0], pred[:, 1], ".-", c="blue", label="pred", linewidth=1.0, zorder=100)
            axs[row, col].plot(
                pred_init[:, 0], pred_init[:, 1], ".-", c="gray", label="pred_init", linewidth=0.5, alpha=0.5
            )

            axs[row, col].scatter(pred[idx, 0], pred[idx, 1], s=scatter_size, c="cyan", marker="X")
            axs[row, col].scatter(pred[idx + 1, 0], pred[idx + 1, 1], s=scatter_size, c="cyan", marker="X")
            axs[row, col].plot(
                action_pick[:, 0], action_pick[:, 1], "o-", label="action_pick", linewidth=3, color="red"
            )
            axs[row, col].plot(
                action_place[:, 0], action_place[:, 1], "o-", label="action_place", linewidth=3, color="green"
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


#########################################


class ActionMaxBending:
    def __init__(self, model, num_steps=500, lr=1e-3, verbose=False, device="cpu"):
        self.model = model
        self.num_steps = num_steps
        self.lr = lr
        self.verbose = verbose
        self.device = device

    def loss_fcn(self, x):
        """
        Loss function for maximizing the angle between the two edges. Works for batched inputs.
        """
        bs = x.shape[0]
        num_nodes = x.shape[1]

        # compute point to point vectors
        diff = x[:, 1:, :] - x[:, :-1, :]

        # move to 3D
        diff = torch.cat((diff, torch.zeros((bs, num_nodes - 1, 1), dtype=torch.float32, device=self.device)), dim=-1)

        # padding i+1 - i
        diff_pos = torch.cat((diff, torch.zeros((bs, 1, 3), dtype=torch.float32, device=self.device)), dim=1)

        # padding i - i-1
        diff_pos_p = torch.cat((torch.zeros((bs, 1, 3), dtype=torch.float32, device=self.device), diff), dim=1)

        # beta numerator
        cross = torch.cross(diff_pos, diff_pos_p)
        beta_num = torch.norm(cross, dim=2)

        # beta denominator
        beta_den = (diff_pos * diff_pos_p).sum(dim=-1).to(torch.float)
        beta_den[:, 0] = torch.ones((bs,), dtype=torch.float, device=self.device)
        beta_den[:, -1] = torch.ones((bs,), dtype=torch.float, device=self.device)

        # beta
        betas = torch.atan(beta_num / beta_den)

        return -torch.sum(betas, dim=1)

    def find_action(self, dlo_0, action, params, print_every=100):
        idx = action[:, 0]
        trainable_x = torch.nn.Parameter(action[:, 1].clone(), requires_grad=True)
        trainable_y = torch.nn.Parameter(action[:, 2].clone(), requires_grad=True)
        trainable_theta = torch.nn.Parameter(action[:, 3].clone(), requires_grad=True)
        optimizer = torch.optim.Adam(
            [
                {"params": trainable_x, "lr": self.lr},
                {"params": trainable_y, "lr": self.lr},
                {"params": trainable_theta, "lr": self.lr},
            ],
            lr=self.lr,
        )

        tanh = torch.nn.Tanh()

        opt_log_dict = {}
        for step in range(self.num_steps):
            optimizer.zero_grad()

            dispx = tanh(trainable_x)
            dispy = tanh(trainable_y)
            theta = tanh(trainable_theta)

            action_tn = torch.cat(
                [idx.unsqueeze(1), dispx.unsqueeze(1), dispy.unsqueeze(1), theta.unsqueeze(1)],
                dim=1,
            )

            pred = self.model(dlo_0, action_tn, params)

            losses = self.loss_fcn(pred)

            action_save = action_tn.clone()
            opt_log_dict[step] = {"losses": to_numpy(losses), "action": action_save}

            if self.verbose:
                if step % print_every == 0:
                    print(f"step: {step}, mean_batch_loss: {torch.mean(losses).item()}")

            # backward
            combined_loss = torch.sum(losses)
            combined_loss.backward()
            optimizer.step()

        return opt_log_dict


class ActionMaxGradNorm:
    def __init__(self, model, num_steps=500, lr=1e-3, verbose=False):
        self.model = model
        self.num_steps = num_steps
        self.lr = lr
        self.verbose = verbose

    def loss_fcn(self, x):
        return -torch.linalg.norm(x, axis=-1)

    def find_action(self, dlo_0, action, params, print_every=100):
        idx = action[:, 0]
        trainable_x = torch.nn.Parameter(action[:, 1].clone(), requires_grad=True)
        trainable_y = torch.nn.Parameter(action[:, 2].clone(), requires_grad=True)
        trainable_theta = torch.nn.Parameter(action[:, 3].clone(), requires_grad=True)
        param_kd = torch.nn.Parameter(params[:, 0].clone(), requires_grad=True)
        param_kb = torch.nn.Parameter(params[:, 1].clone(), requires_grad=True)
        optimizer = torch.optim.Adam(
            [
                {"params": trainable_x, "lr": self.lr},
                {"params": trainable_y, "lr": self.lr},
                {"params": trainable_theta, "lr": self.lr},
                {"params": param_kd, "lr": self.lr},
                {"params": param_kb, "lr": self.lr},
            ],
            lr=self.lr,
        )

        tanh = torch.nn.Tanh()

        opt_log_dict = {}
        for step in range(self.num_steps):
            optimizer.zero_grad()

            dispx = tanh(trainable_x)
            dispy = tanh(trainable_y)
            theta = tanh(trainable_theta)

            action_tn = torch.cat(
                [idx.unsqueeze(1), dispx.unsqueeze(1), dispy.unsqueeze(1), theta.unsqueeze(1)],
                dim=1,
            )

            params_tn = torch.cat([param_kd.unsqueeze(1), param_kb.unsqueeze(1), params[:, -1].unsqueeze(1)], dim=1)

            pred = self.model(dlo_0, action_tn, params_tn)

            pred_sum = torch.sum(pred)

            (kd_grad,) = torch.autograd.grad(pred_sum, param_kd, create_graph=True)
            (kb_grad,) = torch.autograd.grad(pred_sum, param_kb, create_graph=True)

            losses = self.loss_fcn(torch.cat([kd_grad.unsqueeze(1), kb_grad.unsqueeze(1)], dim=1))

            action_save = action_tn.clone()
            opt_log_dict[step] = {"losses": to_numpy(losses), "action": action_save}

            if self.verbose:
                if step % print_every == 0:
                    print(f"step: {step}, mean_batch_loss: {torch.mean(losses).item()}")

            # backward
            combined_loss = torch.sum(losses)
            combined_loss.backward()
            optimizer.step()

        return opt_log_dict
