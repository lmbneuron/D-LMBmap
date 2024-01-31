D-LMBMap
========

## Architecture
------
![architecture_img](./imgs/architecture.png)

## Tutorial
-------
**First, please unzip the data.zip. The data.zip contains the data for registration and one saved checkpoint.**
```
unzip data.zip
put the unzip files under the root file so the fold sturct should go like these:
Whole Brain Registration
│   ...
│   
└───data
      │   
      └───────data
      |   
      └───────data_2
      │   
      └───────data_4
      │
      └───────gt
|   ...
```
> ### **Training**
> #### 1/4 training
>```
>example: python main.py --train -c configs/train_4.yaml -o ckp/train_4
>necessary parameter: 
>--train: config the network into training mode
>-c/--config: appoint the config(.yaml) file which contains the training settings
>optional parameter:
>-o/--output: select the folder where training result will be save at
>```
>#### 1/2 training
>First you must set the TrainConfig->checkpoint in the yaml(e.g. train_2.yaml). The main difference of the train_4.yaml and train_2.yaml is the ModelConfig->Scale. Scale must be set to 1 when 1/4 training and set to 2 when 1/2 training.
>```
>example: python main.py --train -c configs/train_2.yaml -o ckp/train_2
>```
> #### Full size training
> The same as the 1/2 training, you must set the TrainConfig->checkpoint in the yaml and remember to set the ModelConfig->Scale to 3. 
> #### Output folder structure
>After training, a folder will be make in the output path. The structure of the folder are listed below (e.g. ckp/train_4):
>```
>ckp/train_4
>│   config.yaml (the appointed yaml file. You can check the config_tempate.yaml to know more information about the configurations.)    
>│
>└───checkpoint
>│   │   *.pth (model checkpoint)
>│   │   ...
>│   
>└───logs
>    │   events.out.tfevents.* (tensorboard file)
>```

-------

> ### Evaluation 
> If you want to have the registration results of data in tot.json, you can run this command.
> We have provided a checkpoint for use which is saved after 1/2 training. (./ckp/checkpoint)

>```
>example: python main.py --eval -c ./ckp/checkpoint/config.yaml --checkpoint ./ckp/checkpoint/checkpoint/7699.pth -o result/eval
>
>necessary parameter:
>--eval: config the network into evaluate mode
>--checkpoint: select the checkpoint(.pth) file using in evaluation
>-c/--config: appoint the config(.yaml) file which contains the evaluate settings, normally it shall be the same as training config corresponding to the checkpoint file
>
>optional parameter:
>-o/--output: select the folder where evaluation result will be save at
>```
> #### Output folder structure
>After evaluation, a folder will be make in the output path. The structure of the folder are listed below (e.g. result/eval):
>```
>result/eval
>│   config.yaml (the appointed yaml file)    
>│   log.txt (the metrics of different constraints which are listed in the config.yaml)    
>└───checkpoint
>│   │   checkpoint.pth (model checkpoint)
>│   
>└───fix (fix images in the tot.json)
>|    └─────e.g. allen_181207_10_39_06
>|        │ allen_181207_10_39_06.tiff
>|        │ allen_181207_10_39_06_tra.tiff
>|        │ allen_181207_10_39_06_hpf.tiff
>|        │ ...
>|    └─────...
>|
>└───mov (moving images in the tot.json)
>|    └─────e.g. allen_181207_10_39_06
>|        │ allen_181207_10_39_06.tiff
>|        │ allen_181207_10_39_06_tra.tiff
>|        │ allen_181207_10_39_06_hpf.tiff
>|        │ ...
>|    └─────...
>|
>└───reg (regstration images)
>|    └─────e.g. allen_181207_10_39_06
>|        │ allen_181207_10_39_06.tiff
>|        │ allen_181207_10_39_06_tra.tiff
>|        │ allen_181207_10_39_06_hpf.tiff
>|        │ ...
>|    └─────...
>```

>### Registration
>If you have the ground truth masks or neuron structures (e.g. axon, soma), you can use this command to deform the masks when registration. The function of this command is similar with the evaluation command.
>
>```
>example: python main.py --test -c ./ckp/checkpoint/config.yaml --checkpoint ./ckp/checkpoint/checkpoint/7699.pth --test_config configs/test_config_template.yaml -o result/test --upsample 2
>
>necessary parameter:
>--test: config the network into register mode
>--checkpoint:select the checkpoint(.pth) file using in registration
>-c/--config:appoint the config(.yaml) file, normally it shall be the same as training config corresponding to the checkpoint file
>--test_config:appoint the config(.yaml) file which contains the information of origin size image
>
>optional parameter:
>-o/--output:select the folder where registration result will be save at
>--updample: size of upsampling. For example, the ./ckp/checkpoint/ is trained on the 1/2 size images (160*228*264), and the data in the test_config is the original size (320*456*528), so we must set the --upsample to 2.
>```
> #### Output folder structure
>After registration, a folder will be make in the output path. The structure of the folder are listed below (e.g. result/test):
>```
>result/test
>│   config.yaml (the appointed yaml file)    
>│   test_config.yaml (the appointed yaml file by --test_config)    
>│   log.txt (the metrics of different constraints which are listed in the config.yaml)    
>│   result.csv (the metrics of different constraints which are listed in the config.yaml)    
>└───fix (fix images in the test_config.yaml/TrainConfig/data->tot.json)
>|    └─────e.g. allen_181207_10_39_06
>|        │ allen_181207_10_39_06.tiff
>|        │ allen_181207_10_39_06_tra.tiff
>|        │ allen_181207_10_39_06_hpf.tiff
>|        │ ...
>|    └─────...
>|
>└───mov (moving images in the test_config.yaml/TrainConfig/data->tot.json)
>|    └─────e.g. allen_181207_10_39_06
>|        │ allen_181207_10_39_06.tiff
>|        │ allen_181207_10_39_06_tra.tiff
>|        │ allen_181207_10_39_06_hpf.tiff
>|        │ ...
>|    └─────...
>|
>└───reg (regstration images test_config.yaml/TrainConfig/data->tot.json)
>|    └─────e.g. allen_181207_10_39_06
>|        │ allen_181207_10_39_06.tiff
>|        │ allen_181207_10_39_06_tra.tiff
>|        │ allen_181207_10_39_06_hpf.tiff
>|        │ ...
>|    └─────...
>```


## Other Tools
>```
>ms_regnet
>└───preprocess
>│   │   writejson.py (generate the train.json, test.json and tot.json)
>│   │   to use your own data, run this script to generate setting files, make sure your data struct is similar to the example data.
>│   
>└───tools
>    │   changeresolution.py (downsample the images in one folder, you can make the data_2 or data_4 by data.)
>```