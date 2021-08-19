"""Graph embedding"""

import torch_geometric as tg
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def speye(n, dtype=torch.float):
    """identity matrix of dimension n as sparse_coo_tensor."""
    return torch.sparse_coo_tensor(torch.tile(torch.arange(n, dtype=torch.long), (2, 1)),
                                   torch.ones(n, dtype=dtype),
                                   (n, n))


class DistanceDecoder(torch.nn.Module):
    """
    implements the distance decoder which predicts the probability of an edge as the exponential of the
    negative euclidean distance between nodes
    """
    def __init__(self):
        super(DistanceDecoder, self).__init__()
        self.dist = torch.nn.PairwiseDistance()

    def forward(self, z, edge_index, sigmoid=True):
        """
        compute decoder values

        Args:
            z: input coordinates
            edge_index: edges
            sigmoid: if ``True``, return exponential of negative distance, else return negative distance

        """
        value = -self.dist(z[edge_index[0]], z[edge_index[1]])
        return torch.exp(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        """
        compute value for all node pairs

        Args:
            z: input coordinates
            sigmoid: if ``True``, return exponential of negative distance, else return negative distance

        """
        adj = -torch.cdist(z, z)
        return torch.exp(adj) if sigmoid else adj


class GAEconv(torch.nn.Module):
    """
    implements the convolution operator for use with :class:`tg.nn.GAE`
    """
    def __init__(self, dim, num_node_features, hidden_dim=32, cached=True, bias=True, add_self_loops=True, normalize=True):
        """
        Initialise parameters

        Args:
            dim: output dimension
            num_node_features: input dimension
            hidden_dim: hidden dimension
            cached: if ``True``, cache the normalised adjacency matrix after first call
            bias: if ``True``, include bias terms
            add_self_loops: if ``True``, add self loops before normalising
            normalize: if ``True``, normalise adjacency matrix
        """
        super().__init__()
        self.conv1 = tg.nn.GCNConv(num_node_features, hidden_dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)
        self.conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)

    def forward(self, data):
        """compute coordinates given data"""
        edge_index = data.edge_index
        x = F.relu(self.conv1(data.x, edge_index))
        return self.conv2(x, edge_index)


class VGAEconv(torch.nn.Module):
    """
    implements the convolution operator for use with :class:`torch_geometric.nn.VGAE`
    """
    def __init__(self, dim, num_node_features, hidden_dim=32, cached=True, bias=True, add_self_loops=True, normalize=True):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(num_node_features, hidden_dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                   normalize=normalize)
        self.mean_conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                        normalize=normalize)
        self.var_conv2 = tg.nn.GCNConv(hidden_dim, dim, cached=cached, bias=bias, add_self_loops=add_self_loops,
                                       normalize=normalize)

    def forward(self, data: tg.data.Data):
        """
        compute mean and variance given data
        Args:
            data: input data

        Returns:
            mu, sigma

        """
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        mu = self.mean_conv2(x, edge_index)
        sigma = self.var_conv2(x, edge_index)
        return mu, sigma


def VGAE_loss(model, data):
    """
    loss function for use with :func:`VGAE_model`

    Args:
        model:
        data:

    Returns:
        loss value
    """
    return model.recon_loss(model.encode(data), data.edge_index) + model.kl_loss() / data.num_nodes


def VGAE_model(dim, hidden_dim, num_features, dist=False):
    """
    initialise a Variational Graph Auto-Encoder model

    Args:
        dim: output dimension
        hidden_dim: inner hidden dimension
        num_features: number of input features
        dist: if ``True`` use distance decoder, otherwise use inner product decoder (default: ``False``)

    Returns:
        initialised :class:`tg.nn.VGAE` model
    """
    if dist:
        return tg.nn.VGAE(encoder=VGAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim),
                          decoder=DistanceDecoder())
    else:
        return tg.nn.VGAE(encoder=VGAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim))


def GAE_loss(model, data):
    """
    loss function for use with :func:`GAE_model`

    Args:
        model:
        data:

    Returns:
        reconstruction loss
    """
    return model.recon_loss(model.encode(data), data.edge_index)


def GAE_model(dim, hidden_dim, num_features, dist=False):
    """
    initialise a Graph Auto-Encoder model

    Args:
        dim: output dimension
        hidden_dim: inner hidden dimension
        num_features: number of input features
        dist: if ``True`` use distance decoder, otherwise use inner product decoder (default: ``False``)

    Returns:
        initialised :class:`tg.nn.GAE` model
    """
    if dist:
        return tg.nn.GAE(encoder=GAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim),
                         decoder=DistanceDecoder())
    else:
        return tg.nn.GAE(encoder=GAEconv(dim, num_node_features=num_features, hidden_dim=hidden_dim))


def lr_grid_search(data, model, loss_fun, validation_loss_fun, lr_grid=(0.1, 0.01, 0.005, 0.001),
                   num_epochs=10, runs=1, verbose=True):
    """
    grid search over learning rate values

    Args:
        data: input data
        model: model to train
        loss_fun: training loss takes model and data as input
        validation_loss_fun: function to compute validation loss input: (model, data)
        lr_grid: learning rate values to try
        num_epochs: number of epochs for training
        runs: number of training runs to average over for selecting best learning rate
        verbose: if ``True``, output training progress

    Returns:
        best learning rate, validation loss for all runs
    """
    val_loss = torch.zeros((len(lr_grid), runs))
    val_start = torch.zeros((len(lr_grid), runs))
    for i, lr in enumerate(lr_grid):
        for r in range(runs):
            model.reset_parameters()
            model = train(data, model, loss_fun, num_epochs=num_epochs, lr=lr, verbose=verbose)
            val_loss[i, r] = validation_loss_fun(model, data)
    model.reset_parameters()
    return lr_grid[torch.argmax(torch.mean(val_loss, 1))], val_loss


def train(data, model, loss_fun, num_epochs=100, verbose=True, lr=0.01, logger=lambda loss: None):
    """
    train an embedding model

    Args:
        data: network data
        model: embedding auto-encoder model
        loss_fun: loss function to use with model (takes arguments ``model``, ``data``)
        num_epochs: number of training epochs
        verbose: if ``True``, display training progress (default: ``True``)
        lr: learining rate (default: 0.01)
        logger: function that receives the training loss as input and is called after each epoch (does nothing by default)

    Returns:
        trained model

    This function uses the Adam optimizer for training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fun(model, data)
        loss.backward()
        optimizer.step()
        logger(float(loss))
        if verbose:
            print(f'epoch {e}: loss={loss.item()}')
        # schedule.step()
    return model


def embedding(model, data):
    """
    Compute embedding for model and data

    Args:
        model: input model
        data: network

    Returns:
        embedding coords for nodes

    This function switches the model to eval state before computing the embedding and restores the original
    training state of the model

    """
    train_state = model.training
    model.training = False
    with torch.no_grad():
        coords = model.encode(data)
    model.training = train_state
    return coords


def reconstruction_auc(coordinates, data, neg_edges=None, dist=False):
    """
    Compute the network reconstruction auc score

    Args:
        coordinates (torch.tensor): embedding to evaluate
        data (tg.utils.data.Data): network data
        neg_edges: edge index for negative edges (optional)
        dist: if ``True``, use distance decoder to evaluate embedding, otherwise use inner-product decoder
              (default: ``False``)

    Returns:
        ROC-AUC for correctly classifying true edges versus non-edges

    By default the function samples the same number of non-edges as there are true edges, such that a score of 0.5
    corresponds to random classification.

    """
    decoder = DistanceDecoder() if dist else tg.nn.InnerProductDecoder()
    if neg_edges is None:
        neg_edges = tg.utils.negative_sampling(data.edge_index, num_nodes=data.num_nodes)
    with torch.no_grad():
        z = torch.cat((decoder(coordinates, data.edge_index, sigmoid=True),
                       decoder(coordinates, neg_edges, sigmoid=True)),
                      dim=0).cpu().numpy()
        y = torch.cat((torch.ones(data.edge_index.shape[1], device='cpu'),
                       torch.zeros(neg_edges.shape[1], device='cpu')),
                      dim=0).numpy()
    return roc_auc_score(y, z)
