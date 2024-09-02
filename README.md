# MCS_Net

This repository is the official implementation of PMS-Net : Rethinking the Light-weight U-shape network from a Convolutional Attention Perspective for Medical Image Segmentation using PyTorch.

![PMS-Net](figure/model.png)



## Main Environments

- python 3.9
- pytorch 2.1.0
- torchvision 0.16.0



## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```



## Prepare the dataset.

- The GlaS  datasets, can be found here ([link](https://academictorrents.com/details/208814dd113c2b0a242e74e832ccac28fcff74e5)), The MoNuSeg  datasets, can be found here ([link](https://monuseg.grand-challenge.org/Home/)), The CHASE datasets, can be found here ([link](https://link.zhihu.com/?target=https%3A//blogs.kingston.ac.uk/retinal/chasedb1/)),The BUSI datasets, can be found here ([link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)), The Chest Xray datasets, can be found here ([link](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset)), The ISIC2017 datasets, can be found here ([link](https://challenge.isic-archive.com/data/#2017)), divided into a 7:1:2 ratio.


- Then prepare the datasets in the following format for easy use of the code:

```
├── datasets
    ├── GlaS
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    ├── MoNuSeg
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    ├── CHASE
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    ├── BUSI
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    ├── Chest Xray
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── ISIC2017
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol 
         
```



## Train the Model

First, modify the model, dataset and training hyperparameters (including learning rate, batch size img size and optimizer etc) in `Config.py`

Then simply run the training code.

```
python3 train_model.py
```



## Evaluate the Model

#### 2. Test the Model

Please make sure the right model, dataset and hyperparameters setting  is selected in `Config.py`. 

Then change the test_session in `Config.py` .

Then simply run the evaluation code.

```
python3 test_model.py
```



## Reference

- [UNet3+](https://github.com/ZJUGiveLab/UNet-Version)

- [MultiResUNet](https://github.com/makifozkanoglu/MultiResUNet-PyTorch)

- [TransUNet](https://github.com/Beckschen/TransUNet)

- [UCTransNet](https://github.com/McGregorWwww/UCTransNet)

- [ACC_UNet](https://github.com/qubvel/segmentation_models.pytorch)

- [MEW_UNet](https://github.com/JCruan519/MEW-UNet)

- [MISSFormer](https://github.com/ZhifangDeng/MISSFormer)

- [U2Net](https://github.com/NathanUA/U-2-Net)

- [SETR](https://github.com/fudan-zvg/SETR)

- [DAEFormer](https://github.com/xmindflow/DAEFormer)

- [AttUNet](https://github.com/ozan-oktay/Attention-Gated-Networks)

- [D-LKANet](https://github.com/xmindflow/deformableLKA)

- [UCTransNet]( https://github.com/McGregorWwww/UDTransNet)

- [ScaleFormer](https://github.com/ZJUGiveLab/ScaleFormer)

- [MTUNet](https://github.com/Dootmaan/MT-UNet)

- [TransCeption](https://github.com/xmindflow/TransCeption)

  ​


## Contact

Yang:(1258595425yyw@gmail.com)
