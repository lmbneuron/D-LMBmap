Torch 1.5.1 & torchvision 0.6.1
Python 3.7.3

###### 1. Prepare data
- The data of the coronal plane is placed under the `train_dataset` folder. 

- The directory structure of the data is as follows: 


```
├── tranin_data
│   ├──brain1: experimental brain
│   │       ├──0001.tiff: brain slice
│   │       └──...
│   ├──allen: Allen atlas CCFv3
│   │       ├──0001.tiff
│   │       └──...
│   ├──...
```

###### 2. Build data txt for brain1 and allen respectively
-

- Run the following python file to build the data txt (both brain1 and allen need to run this file)

```
python build_dataset.py
```

###### 3. Training/Testing

- Run the training file to train the style transfer model (change the parameters in `options/base_options.py`). The model is saved in `checkpoints` folder.

```
python train.py
```

- Run the testing file to test the data. The output is saved in `results` folder.
```
python test.py --model cycle_gan
```

```
├── results
│   ├──model1: experimental brain
│   │       ├──0001_real_A.tiff: original brain1's image
│   │       └──0001_real_B.tiff: original allen's image
│   │       └──0001_fake_A.tiff: allen image with bain1's style
│   │       └──0001_real_B.tiff: brain1's image with allen style (style transferred results)
│   │       └──0001_rec_A.tiff: reconstructed brain1's image
│   │       └──0001_rec_B.tiff: reconstructed allen's image
```

- Run the union file to combine the 2d data to 3d. The output is saved in `results` folder.

```
python union.py
```

- Run the eval file to compute the SSIM, PNSR, and FID. 

```
python eval.py 