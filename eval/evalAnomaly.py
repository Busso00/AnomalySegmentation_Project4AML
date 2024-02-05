# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from transform import Relabel, ToLabel, Colorize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


input_transform = Compose([ToTensor()])

target_transform = Compose([ToLabel()])


def get_logit(network, image, transform=None, as_numpy=True):
    if transform is None:
        transform = input_transform
    x = transform(image)
    x = x.unsqueeze(0).cuda()
  
    with torch.no_grad():
        y = network(x)
    
    if as_numpy:
        y = y.data.cpu().numpy()[0].astype("float32")

    return y

def get_softmax(network, image, transform=None, as_numpy=True):
    if transform is None:
        transform = input_transform
    x = transform(image)
    x = x.unsqueeze(0).cuda()
  
    with torch.no_grad():
        y = network(x)
    probs = F.softmax(y, 1)
    if as_numpy:
        probs = probs.data.cpu().numpy()[0].astype("float32")
        
    return probs

def get_softmax_t(network, image, temp, transform=None, as_numpy=True):
    if transform is None:
        transform = input_transform
    x = transform(image)
    x = x.unsqueeze(0).cuda()
  
    with torch.no_grad():
        y = network(x)
        y = torch.div(y, temp )
    probs = F.softmax(y, 1)
    if as_numpy:
        probs = probs.data.cpu().numpy()[0].astype("float32")
        
    return probs

def get_entropy(network, image, transform=None, as_numpy=True):
    probs = get_softmax(network, image, transform, as_numpy=False)
    entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=1), torch.log(torch.tensor(probs.shape[1])))
    if as_numpy:
        entropy = entropy.data.cpu().numpy()[0].astype("float32")
    return entropy

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="/content/drive/MyDrive/ProjectAML/save/erfnet_train_cityscape/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="/content/drive/MyDrive/ProjectAML/eval/erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    #change to data path
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--anomalyScore', default='maxLogit')
    parser.add_argument('--temp', default=1.0)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    temp = float(args.temp)
    anomalyScore = args.anomalyScore
    modelpath = args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in args.input:
        
        images = Image.open(path).convert('RGB')#).resize((1024,512), resample=Image.BILINEAR)
        
        
         
        if anomalyScore == 'MSP':
          probs = get_softmax(model, images)
          anomaly_result = 1 - np.max(probs, axis=0)
        elif anomalyScore == 'maxLogit':
          logit = get_logit(model, images)
          anomaly_result = - np.max(logit, axis=0)
        elif anomalyScore == 'maxEntropy':
          anomaly_result = get_entropy(model, images)
        elif anomalyScore == 'tempMSP':
          probs = get_softmax_t(model, images, temp)
          anomaly_result = 1 - np.max(probs,axis=0)
        
           

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
          pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
          pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
          pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask).astype('int32')

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts) # void value = 2 -> mapped to anomaly-not-obstacle
        
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts) 
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((201>ood_gts>1), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:

            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        #del anomaly_result, ood_gts, mask
        #torch.cuda.empty_cache()

        
    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    #only map to in/out distribution
    
    ood_mask = (ood_gts == 1) #anomaly
    ind_mask = (ood_gts == 0) #not anomaly, in distribution

    ood_out = anomaly_scores[ood_mask] #select only certain anomaly scores (output) (corresponding to ood_mask)
    ind_out = anomaly_scores[ind_mask] #select only certain anomaly scores (output) (corresponding to ind_mask)

    #print(ind_mask.shape)
    #print(ood_mask.shape)
    #print(ind_out.shape)
    #print(ood_out.shape)
    #print(anomaly_scores.shape)
   
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    
    prc_auc = average_precision_score(val_label, val_out)
    #in practice set a threshold on value in order to truly classify as anomaly 95% of samples
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()