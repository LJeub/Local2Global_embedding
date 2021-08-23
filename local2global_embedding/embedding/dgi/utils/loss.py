import torch_geometric as tg
import torch


class DGILoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fun = torch.nn.BCEWithLogitsLoss()

    def forward(self, model, data: tg.data.Data):
        device = data.edge_index.device
        nb_nodes = data.num_nodes
        idx = torch.randperm(nb_nodes, device=device)

        shuf_fts = data.x[idx, :]

        lbl_1 = torch.ones(nb_nodes)
        lbl_2 = torch.zeros(nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 0)

        logits = model(data.x, shuf_fts, data.edge_index, None, None, None)

        return self.loss_fun(logits, lbl)
