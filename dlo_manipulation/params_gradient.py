import torch, copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dlo_manipulation.model import FCMul, EarlyStopping
from dlo_manipulation.dataset import DloSample


# nn_type = "fcmul", "rbf", "in_bilstm", "bilstm"


class ParametersFinderGradient:
    def __init__(self, checkpoint_path, device="cpu", lr=1e-3, opt_epochs=500, verbose=False, nn_type="fcmul"):
        self.device = device
        self.lr = lr
        self.opt_epochs = opt_epochs
        self.verbose = verbose

        # LOAD CHECKPOINT
        state = torch.load(checkpoint_path, map_location=torch.device(self.device))

        # MODEL
        self.nn_type = nn_type
        if nn_type == "fcmul":
            self.model = FCMul(n_pts=state["num_points"], pts_dim=state["dim_points"], hidden_dim=state["hidden_dim"])
        else:
            raise NotImplementedError(f"nn_type {nn_type} not implemented")

        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()

        # LOSS
        self.loss_fcn = lambda x, y: torch.mean(torch.linalg.norm(x - y, axis=-1))

        self.sample_obj = DloSample(state["parameter_ranges"])

    def load_sample(self, file_path):
        dlo_0_n, dlo_1_n, action_n, _ = self.sample_obj.load_and_normalize_sample(file_path)

        dlo_0 = torch.from_numpy(dlo_0_n.copy()).float()
        dlo_1 = torch.from_numpy(dlo_1_n.copy()).float()
        action = torch.from_numpy(action_n.copy()).float()
        return dlo_0, dlo_1, action

    def normalize_params(self, params):
        return self.sample_obj.normalize_params(params)

    def run(self, dataset, mass=0.05, kd_init_n=0.5, kb_init_n=0.5):
        mass_normalized = self.sample_obj.dlo_params.normalize_mass(mass)

        print(f"input mass: {mass}, normalized mass: {mass_normalized}")

        batch_size = len(dataset)

        ##############
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        trainable_params = torch.nn.Parameter(
            torch.Tensor(np.array([kd_init_n, kb_init_n])[np.newaxis]), requires_grad=True
        )
        mass_nt = torch.Tensor(np.array([mass_normalized])[np.newaxis])

        optimizer = torch.optim.Adam([trainable_params], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.1)

        sigmoid = torch.nn.Sigmoid()

        best_loss = np.inf
        best_params_n = None
        list_losses = []
        list_trainable_params = []
        list_lr = []

        early_stopping = EarlyStopping(patience=50, min_epochs=200)
        for epoch in tqdm(range(self.opt_epochs)):
            epoch_loss = 0.0
            for i, data in enumerate(loader):
                optimizer.zero_grad()

                dlo_0, dlo_1, action = data

                train_kd = sigmoid(trainable_params[:, 0])
                train_kb = sigmoid(trainable_params[:, 1])

                params = torch.cat([train_kd.unsqueeze(0), train_kb.unsqueeze(0), mass_nt], dim=1)
                params = params.tile([batch_size, 1])

                pred = self.model(dlo_0, action, params)
                l = self.loss_fcn(pred, dlo_1)

                l.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += l.item()

            scheduler.step(epoch_loss)
            epoch_loss /= len(loader)

            # EARLY STOPPING
            if early_stopping.stop(epoch_loss):
                print("Early Stopping!")
                break

            if self.verbose:
                if epoch % 50 == 0:
                    print(
                        f"epoch {epoch + 1}/{self.opt_epochs}: "
                        f"loss {epoch_loss:.5f}, param_kd: {train_kd[0].item():.5f}, "
                        f"param_kb: {train_kb[0].item():.5f}, "
                    )

            # save values for logging

            list_trainable_params.append([train_kd.item(), train_kb.item()])
            list_losses.append(epoch_loss)
            list_lr.append(optimizer.param_groups[0]["lr"])

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_params_n = copy.deepcopy([train_kd.item(), train_kb.item()])

        list_trainable_params = np.array(list_trainable_params)
        kds = list_trainable_params[:, 0]
        kbs = list_trainable_params[:, 1]

        kds_denormalized = [self.sample_obj.dlo_params.denormalize_damping(x) for x in kds]
        kbs_denormalized = [self.sample_obj.dlo_params.denormalize_bending(x) for x in kbs]

        # BEST
        best_kd_denormalized = self.sample_obj.dlo_params.denormalize_damping(best_params_n[0])
        best_kb_denormalized = self.sample_obj.dlo_params.denormalize_bending(best_params_n[1])

        if self.verbose:
            print(f"PARAMS: [{kds[-1]:.5f}, {kbs[-1]:.5f}]")

        log_dict = {
            "kds_normalized": kds,
            "kbs_normalized": kbs,
            "kds": kds_denormalized,
            "kbs": kbs_denormalized,
            "losses": list_losses,
            "lr": list_lr,
            "train_values": list_trainable_params,
            "mass": mass,
            "mass_normalized": mass_normalized,
        }

        return (best_kd_denormalized, best_kb_denormalized), log_dict

    def plot_result(self, logs, save_path=None):
        kds = logs["kds"]
        kbs = logs["kbs"]
        losses = logs["losses"]
        lr = logs["lr"]

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        x = np.arange(len(kds))
        axs[0].plot(x, kds)
        axs[0].set_title("KD Curve")
        axs[0].set_ylabel("KD")
        axs[0].set_xlabel("Epoch")
        axs[1].plot(x, kbs)
        axs[1].set_title("KB Curve")
        axs[1].set_ylabel("KB")
        axs[1].set_xlabel("Epoch")
        axs[2].plot(x, losses)
        axs[2].set_title("Loss Curve")
        axs[2].set_ylabel("Loss")
        axs[2].set_xlabel("Epoch")
        axs[3].plot(x, lr)
        axs[3].set_title("LR Curve")
        axs[3].set_ylabel("LR")
        axs[3].set_xlabel("Epoch")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()
