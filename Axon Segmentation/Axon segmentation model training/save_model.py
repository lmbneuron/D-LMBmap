import torch

# path of the model that will be used for whole brain axon segmentation
pre_trained = "/media/myname/DATASET/trained_models/nnUNet/3d_fullres/" \
              "TaskXXX_MYTASKNAME/TRAINER_CLASS_NAME__PLANS_FILE_NAME/fold_FOLD/model_final_checkpoint.model"
pretrain_encoder = torch.load(pre_trained)
torch.save(pretrain_encoder['state_dict'], '../Axonal-semantic-segmenter/example_checkpoint.pth')
