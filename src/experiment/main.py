from src.model.MCT import MCTSys
from torchsummary import summary
import torch

if __name__ == '__main__':
    """
    We provide printing and inference code for the MCT-Grasp, 
    and the complete code will be comming soon
    """
    model = MCTSys(in_chans=4, img_size=224, embed_dim=24,num_heads=[1, 2, 4, 8]).cuda()
    summary(model, (4, 224, 224))
    model = model.cuda()
    image = torch.rand(1, 4, 224, 224).cuda()
    predicts = model(image)
    print(predicts[0].shape)