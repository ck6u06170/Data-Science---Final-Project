import numpy as np
class config(object):
    def __init__(self):
        self.train_data_path= "/home/ma/Pytorch-Medical-Segmentation/data/DS_Train/Train_Images/"
        self.valid_data_path= ""
        self.edg_path       = "/home/ma/Pytorch-Medical-Segmentation/data/DS_Train/gray/" #edge of all the data
        self.label_path     = "/home/ma/Pytorch-Medical-Segmentation/data/DS_Train/Train_Labels/" #ground truth of all the data
        #DS_Public or DS_Private
        self.test_data_path = "/home/ma/Pytorch-Medical-Segmentation/data/DS_Private/"
        self.save_path      = 'model_0608'
        self.model_path     = 'model_0608/model-95.hdf5'
        self.train          = 1 #train:1 test:0
        self.cutoff         = 0.5 #the cutoff of the prediction map

        self.epoches        = 200
        self.batch_size     = 2
          

        self.optim_conf     = {
        'learning_rate':0.0001,
        'weight_decay':0.0001,
        'betas':(0.9, 0.999)
        }
        self.lr_scheduler   = {
        'gamma':0.96
        }
