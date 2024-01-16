import torch, os
import matplotlib.pyplot as plt
import numpy as np

from dlo_manipulation.model import FCMul
from dlo_manipulation.dataset import DloDataset


MAIN_DIR = os.path.join(os.path.dirname(__file__), "..")

PARAMS = np.array([0.2, 0.4, 0.02])[np.newaxis]  # kb, kd, mass

DATA_PATH = os.path.join(MAIN_DIR, "dataset/val")
CHECKPOINT_PATH = "fcmul.pth"

state = torch.load(CHECKPOINT_PATH)
###################################
print("*" * 20)
for k, v in state.items():
    if k != "model":
        print(f"\t{k}: {v}")
print("*" * 20)
###################################

#
parameter_ranges = state["parameter_ranges"]

# MODEL
model = FCMul(n_pts=state["num_points"], pts_dim=state["dim_points"], hidden_dim=state["hidden_dim"])
model.load_state_dict(state["model"])

loss_fcn = lambda x, y: torch.mean(torch.linalg.norm(x - y, axis=-1))

params_fixed = torch.from_numpy(PARAMS).float()


##############################

dataset = DloDataset(DATA_PATH, parameter_ranges)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for i, data in enumerate(loader):
    dlo_0, dlo_1, action, params = data
    pred = model(dlo_0, action, params_fixed)
    loss = loss_fcn(pred, dlo_1).item()

    dlo_0 = dlo_0.squeeze().detach().numpy()
    dlo_1 = dlo_1.squeeze().detach().numpy()
    pred = pred.squeeze().detach().numpy()

    plt.plot(pred[:, 0], pred[:, 1], "o-", label="predicted (NN)")
    plt.plot(dlo_1[:, 0], dlo_1[:, 1], "o-", label="desired")
    plt.plot(dlo_0[:, 0], dlo_0[:, 1], "o-", label="init")

    ax = plt.gca()
    ax.set_title(f"Error NN {(loss*1000):.2f} [mm]", fontsize=10)
    ax.axis("equal")

    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()
    plt.close()
