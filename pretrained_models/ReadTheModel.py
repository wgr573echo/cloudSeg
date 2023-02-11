import torchvision.models as models
import torch
import torchsummary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('hrnetv2_w48_imagenet_pretrained.pth')
model = model.to(device)
torchsummary.summary(model,3,256,256)
# pre_dict = model.state_dict()
# for k, v in pre_dict.items():
#     print (k)