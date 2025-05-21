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


#### [Recommended] Installation Steps
install required dependencies:

# 💡 Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

main/: Contains the scripts for running the experiments on class-imbalance, topology-imbalance and intertwined class and topology imbalance.

Split/: Contains the data preprocessing methods, including:

Topology Imbalance: Methods for handling topology imbalance in graph data.

intertwined class and topology imbalance: Methods for addressing intertwined imbalance scenarios.

To reproduce results in Table 1, please run the following code:
```linux
bash Class.sh
```
To reproduce results in Table 2, please run the following code:
```linux
bash Topology.sh
```
