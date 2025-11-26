# PMS_Net

This repository is the official implementation of PMS-Net: A Lightweight Convolutional Attention U-Shaped Network for Medical Image Segmentation using PyTorch.

![PMS-Net](figure/model.jpg)



## Main Environments

- python 3.9
- pytorch 2.1.0
- torchvision 0.16.0
- For generating GradCAM results, please follow the code on this [repository](https://github.com/jacobgil/pytorch-grad-cam)


## Requirements

Install from the `requirements.txt` using:

```
pip install -r requirements.txt
```



## Prepare the dataset.

- The GlaS  datasets, can be found here ([Google Drive](https://drive.google.com/file/d/1F3MA4xUWRMqXLweGu3Sw6SdA1HVX-5y6/view?usp=drive_link))
- The MoNuSeg  datasets, can be found here ([Google Drive](https://drive.google.com/file/d/1MNW82SC5ldNIwjD1vs4m9Dt0u_HtI2-b/view?usp=drive_link))
- The CHASE datasets, can be found here ([Google Drive](https://drive.google.com/file/d/1bq6xZRaxxF-RLF8U7vLCeK5e3mwZtkUf/view?usp=drive_link))
- The BUSI datasets, can be found here ([Google Drive](https://drive.google.com/file/d/1ZeiUCygaXPqIdzsHfZeOIQ4cWgOTBJev/view?usp=drive_link)),
- The Chest Xray datasets, can be found here ([Google Drive](https://drive.google.com/file/d/1O0By-whbMNFQYYUQk5OHgDuZohhO0k7O/view?usp=drive_link))
- The ISIC2017 datasets, can be found here ([Google Drive](https://drive.google.com/file/d/1h-DnH5u2VxNLz3k4Bz8p_r5X5rQO7aHZ/view?usp=drive_link))
- The Synapse datasets, can be found ([Google Drive](https://drive.google.com/file/d/1c2oK6-LeUtrOdVNpbft5dpbatRoFrBIm/view?usp=drive_link))



#### Then prepare the datasets in the following format for easy use of the code:

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

- [TransUNet](https://github.com/Beckschen/TransUNet)

- [UCTransNet](https://github.com/McGregorWwww/UCTransNet)

- [ACC_UNet](https://github.com/qubvel/segmentation_models.pytorch)

- [UNeXt](https://github.com/jeya-maria-jose/UNeXt-pytorch)

- [DAEFormer](https://github.com/xmindflow/DAEFormer)

- [D-LKANet](https://github.com/xmindflow/deformableLKA)

- [UDTransNet]( https://github.com/McGregorWwww/UDTransNet)

- [TransCeption](https://github.com/xmindflow/TransCeption)

  ​

