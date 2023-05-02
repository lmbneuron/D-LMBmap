Torch 1.5.1 & torchvision 0.6.1
Python 3.7.3

###### 1. Prepare data
- The data of the coronal plane is placed under the `data/train_dataset_view1` folder. 

  The data of horizontal plane is placed under the `data/train_dataset_view2` folder.
  
  The data can be obtained from either coronal, horrizontal, and sagittal views.

- The directory structure of the data is as follows: 


```
├── data
│   ├──train_dataset_view1
│   │       └──data_brain1: in view1
│   │              └──0001.tiff
│   │              └──...
│   │       └──mask_NAMA_brain1 (mask_CH_brain1)
│   │              └──0001.ome.nii
│   │              └──...
│   │       └──...
│   ├──train_dataset_view2
│   │       └──data_brain1: in view2
│   │              └──0001.tiff
│   │              └──...
│   │       └──mask_NAME_brain1
│   │              └──0001.ome.nii
│   │              └──...
│   │       └──...
```

###### 2. Build data txt

- Run the python file to build the data txt (for both brain images from view1 and view2)

```
python build_dataset.py
```

###### 3. Train/Test/Eval

- Run the training file to train the model (2 brain regions trained together) (change the parameters in `config.py`).The model is saved in `output` folder.

```
python train.py
```

- Run the test file to test the data.(change the parameters in `config.py`). The output is saved in `output` folder.


```
python test.py
```

- Run the union file to combine the 2d data to 3d (change the parameters in `config.py`). The output is saved in `output` folder.

```
python 2dto3d_nii_coronal.py (coronal plane)
python 2dto3d_nii_horizontal.py (horizontal plane)
```

- Run the eval file to compute the dice score. Mask file is is placed under the `output/mask` folder with 3d. 

```
python eval_indicator.py 
```

###### 