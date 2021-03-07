def build_model(args, MODEL, CUDA_OK=False):
    model = MODEL(**args)
    if CUDA_OK:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
