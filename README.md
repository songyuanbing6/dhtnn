# Double-head Transformer Neural Network for Molecular Property Prediction
This is a Pytorch implementation of Double-head Transformer Neural Network for Molecular Property Prediction (DHTNN)

![Image text](https://github.com/songyuanbing6/DHTNN/blob/master/Figure%201.png)
**Figure 1** Overall DHTNN architectural diagram
## Environment
```
python=3.8.10
pytorch=1.4.0
torchvision=0.5.0
chemprop=1.3.0
flask=2.0.1
hyperopt=0.2.5
matplotlib=3.5.1
numpy=1.20.3
pandas=1.2.4
pandas-flavor=0.2.0
pip=21.1.1
rdkit=2021.03.2
scikit-learn=0.24.2
scipy=1.6.3
tensorboardX=2.2
tqdm=4.61.0
typed-argument-parser=1.6.2
git+https://github.com/bp-kelley/descriptastorus
```
## Dataset
There are six datasets, which are Lipophilicity, PDBbind, PCBA, BACE, Tox21, and SIDER.

You can be get the data by uncompressing `data.zip`. All the data used in the experiments is in here.

## Train and Test the model
```
chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is one of [classification, regression] depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

## Acknowledgement
We refer to the paper of [Analyzing Learned Molecular Representations for Property Prediction](https://arxiv.org/abs/1904.01561). We are grateful for the previous work of swansonk14 team.