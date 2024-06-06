

# erfnet_encoder_pretrained.pth.tar
from https://github.com/Eromera/erfnet
pretrained ImageNet with a classification task then removing last layers
# erfnet_pretrained.pth:
ERFNet full on (19 class of cityscape)
erfnet_pretrainded_20class.pth
ERFNet full on (19 class + background of cityscape)

# enet.pth:
our trained enet
- 50 epochs on 19 + background classes
- wrong due to the fact that i resumed training increasing number of epochs, next time I will use time-based scheduler instead of step-based

- scheduler:
  lr(epoch)=lr0*(1-(epoch)/tot_epoch)**0.9
  
- optimizer = Adam (https://arxiv.org/abs/1412.6980)
  - lr=5e-4, 
  - betas=(0.9, 0.999), #weight for running avg of 1st and 2nd order momentums
  - eps=1e-08, #numerical stability purpose
  - weight_decay=1e-4 #momentum
  
- batch size 6

# our BiSENet training:

- 200 epochs
- ? completed

- scheduler:
  lr(epoch)=lr0*(1-(epoch)/tot_epoch)**0.9

- optimizer = Adam (https://arxiv.org/abs/1412.6980)
  - lr=5e-4, 
  - betas=(0.9, 0.999), #weight for running avg of 1st and 2nd order momentums
  - eps=1e-08, #numerical stability purpose
  - weight_decay=1e-4 #momentum

- batch size 16
