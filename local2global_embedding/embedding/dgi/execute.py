import torch
import torch.nn as nn
import torch_geometric as tg
import argparse


from .models import DGI, LogReg
from .utils.loss import DGILoss


parser = argparse.ArgumentParser(description="DGI test script")
parser.add_argument('--datapath', default='/tmp/cora')
args = parser.parse_args()

dataset = 'cora'

loss_fun = DGILoss()

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu'  # special name to separate parameters

data = tg.datasets.Planetoid(name='Cora', root=args.datapath)[0]
r_sum = data.x.sum(dim=1)
r_sum[r_sum == 0] = 1.0  # avoid division by zero
data.x /= r_sum[:, None]

# adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
# features, _ = process.preprocess_features(features)

nb_nodes = data.num_nodes
ft_size = data.num_features
nb_classes = data.y.max().item() + 1

# adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

# if sparse:
#     sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
# else:
#     adj = (adj + sp.eye(adj.shape[0])).todense()

# features = torch.FloatTensor(features[np.newaxis])
# if not sparse:
#     adj = torch.FloatTensor(adj[np.newaxis])
# labels = torch.FloatTensor(labels[np.newaxis])
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    data.cuda()

xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()
    loss = loss_fun(model, data)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

embeds, _ = model.embed(data.x, data.edge_index, None)
train_embs = embeds[data.train_mask]
val_embs = embeds[data.val_mask]
test_embs = embeds[data.test_mask]
#
train_lbls = data.y[data.train_mask]
val_lbls = data.y[data.val_mask]
test_lbls = data.y[data.test_mask]

tot = torch.zeros(1)
# tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    if torch.cuda.is_available():
        log.cuda()
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)



    pat_steps = 0
    if torch.cuda.is_available():
        best_acc = torch.zeros(1, device='cuda')
    else:
        best_acc = torch.zeros(1)
        
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())
