# DGI

This code is adapted from the reference implementation of Deep Graph Infomax (Veličković *et al.*, ICLR 2019): [https://arxiv.org/abs/1809.10341](https://arxiv.org/abs/1809.10341)

![](https://camo.githubusercontent.com/f62a0b987d8a1a140a9f3ba14baf4caa45dfbcad/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f757a783779677761637a76747031302f646565705f67726170685f696e666f6d61782e706e673f7261773d31)

The original reference implementation is available at https://github.com/PetarV-/DGI

## Overview
Here we provide an implementation of Deep Graph Infomax (DGI) in PyTorch that works with data in [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) edge-index format. The repository is organised as follows:
- `models/` contains the implementation of the DGI pipeline (`dgi.py`);
- `layers/` contains the implementation of a GCN layer (`gcn.py`), the averaging readout (`readout.py`), and the bilinear discriminator (`discriminator.py`);
- `utils/` contains the loss function for training (`loss.py`).

## Reference
If you make advantage of DGI in your research, please cite the following in your manuscript:

```
@inproceedings{
velickovic2018deep,
title="{Deep Graph Infomax}",
author={Petar Veli{\v{c}}kovi{\'{c}} and William Fedus and William L. Hamilton and Pietro Li{\`{o}} and Yoshua Bengio and R Devon Hjelm},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=rklz9iAcKQ},
}
```

## License
MIT
