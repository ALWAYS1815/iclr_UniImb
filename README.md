# One for Two: A Unified Framework for Imbalanced Graph Classification via Dynamic Balanced Prototype
We propose UniImb, a novel framework for imbalanced graph classification based on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://www.pyg.org/), to address both types of imbalance in a uniform manner. UniImb first captures multi-scale topology information and introduces differentiable uncertainty through a learnable graph perturbation strategy. It then incorporates dynamic balanced prototype(DBP), which learns representative prototypes from the graph data to enhance representation learning, together with a load-balancing optimization strategy that enforces balanced interaction across imbalanced graph samples. Extensive experiments across 12 datasets and 16 benchmarks demonstrate that UniImb significantly improves graph classification performance under various imbalance settings.


## 📝 Overall architecture of UniImb

<p align="center">
<img src="figs/UniImb.png" width="100%" class="center" alt="logo"/>
</p>

## Installation
You can easily reproduce the results of UniImb by following the steps below:

#### Environment Requirements

Please ensure your Python environment meets  following dependencies:

| Dependency        | Version (≥) |
| ----------------- | ------------|
| Python            | 3.11.11     |
| PyTorch           | 2.3.0       |
| PyTorch-Geometric | 2.6.1       |
| scipy             | 1.15.2      |

#### 💡 Install specific dependencies

```
pip install -r requirements.txt

```

## 🚀 Quick Start 

### `Dataset`

binary classification graph dataset: 
```
MUTAG, PTC-MR, DHFR, PROTEINS, D&D, REDDIT-B, AIDS, NCI1, FRANKENSTEIN
```

multi-class graph dataset: 
```
COLLAB, Synthie, IMDB-MULTI
```

These graph classification datasets are widely used benchmark datasets in Graph Neural Network (GNN) research. They can be automatically downloaded and loaded via [PyTorch Geometric](https://www.pyg.org/), without the need for manual downloading.

### `imbalance type`

Imbalance type:

```
'class', 'topology', 'intertwined class and topology imbalance'
```

### `Imbalance degrees`

Controls the severity of imbalance:

```
'low', 'mid', 'high'
```
### `scripts`

```
📁 main/: Contains the scripts for running the experiments on class-imbalance, topology-imbalance and intertwined class and topology imbalance.

📁 Split/: Contains the data preprocessing methods, including: (1) Topology Imbalance: Methods for handling topology imbalance in graph data. (2) intertwined class and topology imbalance: Methods for addressing intertwined imbalance scenarios.
```
### `Distribution`
In the Dynamic Balanced Prototype (DBP), there are four possible prototype activation distributions, each corresponding to a different optimization loss function. These distributions are provided in the Distribution folder.

```
Zipf, Poisson, Exponential, Uniform
```
Through experiments, we have validated that when the prototype activation distribution is Uniform, it performs the best in handling imbalanced graph classification.

### `backbone`
UniImb performs well across all backbones. We provide three GNNs to choose from, with the best performance observed when using GIN as the backbone.

```
GIN, GCN, GraphSAGE
```

### Run

To reproduce results in Table 1, please run the following code:

```linux
bash Class.sh
```

To reproduce results in Table 2, please run the following code:
```linux
bash Topology.sh
```
