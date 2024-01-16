import numpy as np
import torch


def scale_dlo_actioncentric(dlo, ref_dlo, action):
    ref_dlo_len = torch.sum(torch.linalg.norm(torch.diff(ref_dlo, axis=-2), axis=-1), axis=-1)
    dlo_len = torch.sum(torch.linalg.norm(torch.diff(dlo, axis=-2), axis=-1), axis=-1)
    ratio = ref_dlo_len / dlo_len

    action_ = action.detach().numpy()

    idx = (action_[..., 0] * (dlo.shape[1] - 1.0)).astype(np.int32)

    bs = dlo.shape[0]
    v1 = dlo[np.arange(bs), idx[np.arange(bs)]]
    v2 = dlo[np.arange(bs), idx[np.arange(bs)] + 1]
    action_edge_center = (v1 + v2) / 2.0
    dlo_action_centred = dlo - action_edge_center[:, np.newaxis]
    dlo_action_centred_scaled = dlo_action_centred * ratio[:, None, None]
    dlo_scaled = dlo_action_centred_scaled + action_edge_center[:, np.newaxis]
    return dlo_scaled
