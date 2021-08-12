# Models

All the updated model codes and pretrained models used to benchmark the new NTU-X dataset are provided in this directory.

There are two main sub directories here:

1. <b>model_code</b>: This directory contains the updated codes used to perfrom experiment on the new NTU-X dataset.

2. <b>pretrained_weights</b>: This directory contains the pretrained weights for all the models considered for the experiments.



The main models considered for the benchmarking are:

- [DSTA-Net](https://github.com/lshiwjx/DSTA-Net)
- [MSG3d](https://github.com/kenziyuliu/ms-g3d)
- [PA-ResGCN](https://github.com/yfsong0709/ResGCNv1)
- [4s-ShiftGCN](https://github.com/kchengiva/Shift-GCN)
- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)

(The codes of the models are taken from the repositories of these models hyperlinked in the above list. The codes included in this repo are modified to be trained on NTU-X data)

### Changes made in the code for each model:

#### DSTA-Net, MSg3d, CTR-GCN and ShiftGCN

The only change that is made in the code base of these two models in the new definition of the graphs with the includsion of additional joints of NTU-X dataset. These changes can be seen for DSTA-Net [here](./model_codes/DSTA-Net/graph/ntu_rgb_d.py), for Msg3d  [here](./model_codes/MsG3d/graph/ntu_rgb_d.py), for CTR-GCN [here](./model_codes/CTR-GCN/graph/ntu_rgb_d.py) and for 4s-ShiftGCN [here](./model_codes/4s-ShiftGCN/graph/ntu_rgb_d.py)


#### PA-ResGCNv1

Apart from defining new graph for the additional joints, new parts are also defined for this model since it uses Part segmentation for attention. The new datasets are defined by the name <b>ntu_ax</b>(118 joints) and <b>ntu_hx</b>(67 joints). These changes can be seen [here](./model_codes/PA-ResGCN/src/dataset/graph.py). The config file used for the 67 joints experiments in <b>1013.yaml</b>


Rest of the code structure is same and further details can be found in the READMe of the respective model's repository.

#### How to run the Models

`
python3 main.py --config ./config/nturgbd-cross-subject/train_joint.yaml `

loading pretained models and training for different streams can be handled by changing arguments in the config files as mentioned in the original model's repository.
