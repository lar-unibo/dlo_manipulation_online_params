import torch
import torch.nn as nn
from dlo_manipulation.utils import scale_dlo_actioncentric


class FCMul(nn.Module):
    def __init__(self, n_pts, pts_dim, hidden_dim=256, scale=False):
        super(FCMul, self).__init__()
        self.scale = scale

        self.n_pts = n_pts
        self.pts_dim = pts_dim
        self.N = hidden_dim

        a = nn.ReLU()  # nn.Tanh()
        tanh = nn.Tanh()

        self.dlo = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(self.n_pts * self.pts_dim, self.N),
            a,
            nn.Linear(self.N, self.N),
            a,
            nn.Linear(self.N, self.N),
            a,
        )

        self.act = nn.Sequential(
            nn.Linear(4, self.N),
            a,
            nn.Linear(self.N, self.N),
            a,
        )

        self.param = nn.Sequential(
            nn.Linear(3, self.N),
            tanh,
            nn.Linear(self.N, self.N),
            tanh,
        )

        self.state_action = nn.Sequential(
            nn.Linear(2 * self.N, self.N),
            a,
        )

        self.pred = nn.ModuleList(
            [
                nn.Linear(self.N, self.N),
                a,
                nn.Linear(self.N, self.N),
                a,
                nn.Linear(self.N, self.n_pts * self.pts_dim),
                nn.Unflatten(-1, (self.n_pts, self.pts_dim)),
            ]
        )

    def forward(self, dlo, action, params):
        x_s = self.dlo(dlo)
        x_a = self.act(action)
        x_p = self.param(params)

        x = torch.concat([x_s, x_a], dim=-1)

        x = self.state_action(x)

        for l in self.pred:
            if l._get_name() == "Linear":
                x = x * x_p
            x = l(x)

        x += dlo

        if self.scale:
            x = scale_dlo_actioncentric(x, dlo, action)
        return x


class EarlyStopping:
    def __init__(self, patience=50, min_epochs=100):
        self.patience = patience
        self.min_epochs = min_epochs

        self.no_improve = 0
        self.min_loss = 100000

    def stop(self, loss):
        if loss < self.min_loss:
            self.no_improve = 0
            self.min_loss = loss
        else:
            self.no_improve += 1

        if self.no_improve >= self.patience and self.min_epochs <= self.min_epochs:
            return True
        else:
            return False
