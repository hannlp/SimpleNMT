import torch
import torch.nn as nn

def build_model(args, MODEL, CUDA_OK=False):
    
    # 
    args_dict = {}

    model = MODEL(**args_dict)
    if CUDA_OK:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

print(dir())