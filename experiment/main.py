import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.MCT import MCTSys
from torchsummary import summary
import torch

if __name__ == '__main__':
    """
    We provide the MCT-Grasp network code, 
    and the complete code will be comming soon
    """
    model = MCTSys(in_chans=4, img_size=224, embed_dim=24,num_heads=[2, 2, 8, 2]).cuda()
    summary(model, (4, 224, 224))
    model = model.cuda()
    image = torch.rand(1, 4, 224, 224).cuda()
    predicts = model(image)
    print(predicts[0].shape)