import os, torch, json
import numpy as np
from tqdm import tqdm

from dlo_manipulation.dataset import DloDataset
from dlo_manipulation.model import FCMul

MAIN_DIR = os.path.join(os.path.dirname(__file__), "..")
LOG_INTERVAL = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = dict(
    batch_size=128,
    epochs=100,
    lr=5e-4,
    hidden_dim=256,
    dim_points=2,
    num_points=16,
    dataset_path="dataset",
)


###################################
print(f"Using device: {DEVICE}")
print("*" * 20)
for k, v in config.items():
    print(f"\t{k}: {v}")
print("*" * 20)
###################################

# DATASETS
train_path = os.path.join(MAIN_DIR, config["dataset_path"], "train")
val_path = os.path.join(MAIN_DIR, config["dataset_path"], "val")
parameter_ranges = json.load(open(os.path.join(MAIN_DIR, config["dataset_path"], "parameters.json"), "r"))

train_data = DloDataset(train_path, parameter_ranges)
val_data = DloDataset(val_path, parameter_ranges)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)


print("Train set size: {}".format(len(train_data)))
print("Val set size: {}".format(len(val_data)))
print("")

# MODEL
model = FCMul(n_pts=config["num_points"], pts_dim=config["dim_points"], hidden_dim=config["hidden_dim"])
model = model.to(DEVICE)
loss_fcn = lambda x, y: torch.mean(torch.linalg.norm(x - y, axis=-1))
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


best_loss = np.inf
global_step = 0
for epoch in tqdm(range(config["epochs"])):
    train_epoch_loss = 0.0
    val_epoch_loss = 0.0

    ##############################
    # TRAIN
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        dlo_0, dlo_1, action, params = data
        dlo_0, dlo_1, action, params = dlo_0.to(DEVICE), dlo_1.to(DEVICE), action.to(DEVICE), params.to(DEVICE)

        pred = model(dlo_0, action, params)

        loss = loss_fcn(pred, dlo_1)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()
        global_step += 1

    train_epoch_loss /= len(train_loader)

    ##############################
    # VAL
    model.eval()
    for i, data in enumerate(val_loader):
        dlo_0, dlo_1, action, params = data
        dlo_0, dlo_1, action, params = dlo_0.to(DEVICE), dlo_1.to(DEVICE), action.to(DEVICE), params.to(DEVICE)

        pred = model(dlo_0, action, params)

        loss = loss_fcn(pred, dlo_1)
        val_epoch_loss += loss.item()

    val_epoch_loss /= len(val_loader)

    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        model_path = os.path.join(MAIN_DIR, "best.pth")

        # saving
        state = dict(config)
        state["model"] = model.state_dict()
        state["parameter_ranges"] = parameter_ranges
        torch.save(state, model_path)

    if epoch % LOG_INTERVAL == 0:
        print("Epoch: {}, train loss: {}, val loss: {}".format(epoch, train_epoch_loss, val_epoch_loss))
