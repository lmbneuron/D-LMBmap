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

- `3dto2d_mask.py` is a tool that can convert a 3D label with the suffix .nii (or .tiff) into a 2D image with suffix .ome.nii

  To get 2d mask in `mask_NAME_brain` under `train_dataset_view1` and `train_dataset_view2`, please download 3D mask from [here](https://drive.google.com/file/d/1izj7cyafzgGtg2FPaIMHIf9G6_VQz1Nj/view?usp=sharing). And run `3dto2d_mask.py`  to generate them.

  Before running this script, please make sure the directory structure of the mask is as follows:
  
  ```
  ├──mask_3d
  │  ├──MASK_NAME.nii
  │  ├──...
  ├──data (Automatic creation)
  │  ├──train_dataset_view1
  │  │  ├──MASK_NAME: in view1
  │  │  │   └──0000.ome.nii
  │  │  │   └──...
  │  ├──train_dataset_view2
  │  │  ├──MASK_NAME: in view2
  │  │  │   └──0000.ome.nii
  │  │  │   └──...
  └──3dto2d_mask.py
  ```
  
  A sample command to run the python file is:
  
  ```
  python 3dto2d_mask.py --file_dir .\ --file_3d mask_3d --save_dir .\ --file_2d data --type CH --image_shape 512,320,488
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
