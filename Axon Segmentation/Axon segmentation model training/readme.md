## 0 Preparation
First of all, prepare the environment. 

(1) Install [PyTorch](https://pytorch.org/get-started/locally/) with Python3. You need at least version 1.6.

(2) Install the dependencies with the following commands: 
```
pip install nnunet
pip install -r requirements.txt
```
(3) Our model needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to 
prepare the following directories.

**DATASET/raw_data_base**: This is where our model finds the raw data and stored the cropped data. The folder located at 
DATASET must have at least the subfolder nnUNet_raw_data, which in turn contains one subfolder for each Task. 

**DATASET/preprocessed**: This is the folder where the preprocessed data will be saved. The data will also be read from 
this folder during training. Therefore it is important that it is located on an SSD.

**DATASET/trained_models**: This specifies where our model will save the model weights. If pretrained models are downloaded, this 
is where it will save them.

Then you need to set the environmental variables. There are several ways you can do this. The most common one is to set the paths in your .bashrc file, which is located 
in your home directory. For me, this file is located at /home/myname/.bashrc. You can open it with any text editor of 
choice. If you do not see the file, that may be because it is hidden by default. You can run `ls -al /home/myname` to 
ensure that you see it. In rare cases it may not be present and you can simply create it with `touch /home/myname/.bashrc`.

Once the file is open in a text editor, add the following lines to the bottom:
```
export nnUNet_raw_data_base="/media/myname/DATASET/raw_data_base"
export nnUNet_preprocessed="/media/myname/DATASET/preprocessed"
export RESULTS_FOLDER="/media/myname/DATASET/trained_models"
```
Adapt the paths to your system. Then save and exit. To be save, make sure to reload the .bashrc by running `source /home/myname/.bashrc`. 
You can verify that the paths are set up properly by typing `echo $RESULTS_FOLDER` etc and it should print out the correct folder.

## 1 Training cubes generation and data augmentation
Before this step, make sure you have prepared the training cubes with automatically annotated masks and stored them as below.
Run `create_data.py`, in which base and source directories should be prepared ahead as below. *Note that the number of 
the skeletonized annotations, the automatically annotated masks and the axon cubes should be exactly the same.*
```
└── base(original training data)
　　 └── train
　 　 　　├── cropped-cubes(training axon cubes)
　　  　　│　　└──volume-001.tiff
　　  　　├── Rough-label(automatically annotated masks)
　　 　 　│　　└──label-001.tiff
　　  　　├── Fine-label(skeletonized annotation)
　　 　 　│　　└──label-001.tiff
　　 　 　└── artifacts(junk cubes)
　　 　　　　　└──volume-200.tiff
└── source(data used for histogram matching)
　　 └── train
　 　 　　├── cropped-cubes
　　  　　│　　└──volume-001.tiff
　　  　　├── Rough-label
　　 　 　│　　└──label-001.tiff
　　  　　├── Fine-label
　　 　 　│　　└──label-001.tiff
　　 　 　└── artifacts
　　 　　　　　└──volume-200.tiff
```
We propose three data augmentation methods, histogram matching, cutmix, and local contrast enhancement to augment training data. 
Change the parameters of function `histogram_match_data` in `create_data.py` to choose using histogram matching/cutmix/contrast enhancement or 
not. If you want to use histogram matching, it is better to set both **match_flag** and **join_flag** True so that both 
original cubes and matched cubes can be used for training.
```
cutmix=True  # use cutmix, mix up axon cubes and artifact cubes
match_flag=True, join_flag=True  # use histogram matching, join matched and original cubes
match_flag=True, join_flag=False  # use histogram matching, use only matched cubes
```
## 2 Preprocessing for training axon segmentation model
After step 1 the raw training dataset will be in the folder prepared in step 0 (`DATASET/raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK`, 
where task id `XXX` and task name `MYTASK` are set in `create_data.py`. 

For training our model, a preprocessing procedure is needed. Run this command:
```bash
nnUNet_plan_and_preprocess -t XXX
```
You will find the output in `DATASET/preprocessed/TaskXXX_MYTASK`. 
There are several additional input arguments for this command. Running `-h` will list all of them 
along with a description. If you run out of RAM during preprocessing, you may want to adapt the number of processes 
used with the `-tl` and `-tf` options. The default configuration make use of a GPU with 8 GB memory. Larger memory size 
can be used with options like `-pl3d ExperimentPlanner3D_v21_16GB`.

## 3 Model training
Our model trains all U-Net configurations in a 5-fold cross-validation. This enables the model to determine the 
postprocessing and ensembling (see next step) on the training dataset. 
Training models is done with the `nnUNet_train` command. The general structure of the command is:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD --npz (additional options)
```
CONFIGURATION is a string that identifies the requested U-Net configuration. TASK_NAME_OR_ID specifies what dataset should 
be trained on and FOLD specifies which fold of the 5-fold-cross-validaton is trained. 

TRAINER_CLASS_NAME is the name of the model trainer. To be specific, a normal U-Net will be trained with TRAINER_CLASS_NAME 
`nnUNetTrainerV2`. 
We also propose networks with attention modules. You can use TRAINER_CLASS_NAME `MyTrainerAxial` to train a U-Net with attention modules. 

Out model stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `-c` to the 
training command. Use the `-h` argument for more settings of parameters. 
For FOLD in [0, 1, 2, 3, 4], a sample command is (if `-pl3d ExperimentPlanner3D_v21_16GB` used in step 2): 
```
nnUNet_train 3d_fullres MyTrainerAxial TaskXXX_MYTASK FOLD -p nnUNetPlansv2.1_16GB
```
The trained models will be written to the `DATASET/trained_models/nnUNet` folder. Each training obtains an automatically generated 
output folder name `DATASET/preprocessed/CONFIGURATION/TaskXXX_MYTASKNAME/TRAINER_CLASS_NAME__PLANS_FILE_NAME/FOLD`.
Multi GPU training is not supported.

## 4 Cube Prediction
Once all 5-fold models are trained, use the following command to automatically determine what 
U-Net configuration(s) to use for test set prediction:
```bash
nnUNet_find_best_configuration -m 3d_fullres -t XXX --strict
```
This command will print a string to the terminal with the inference commands you need to use. 
The easiest way to run inference is to simply use these commands. 

For each of the desired configurations(e.g. 3d_fullres), run:
```
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
```
Only specify `--save_npz` if you intend to use ensembling. `--save_npz` will make the command save the softmax 
probabilities alongside of te predicted segmentation masks requiring a lot of disk space. You can also use `-f` to specify 
folder id(s) if not all 5-folds has been trained. 
`--tr` option can be used to specify TRAINER_CLASS_NAME, which should be consistent with the class used in model training. 

A sample command using an U-Net with attention module to generate predictions is: 
`
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t XXX --tr MyTrainerAxial -m 3d_fullres -p nnUNetPlansv2.1_16GB
`

We extract the model weights from the saved checkpoint files(e.g. model_final_checkpoint.model) to `pth` files.
Run `python save_models.py`. The `pth` file will be used for whole brain axon prediction.
