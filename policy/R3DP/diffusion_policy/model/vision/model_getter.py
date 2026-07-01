import torch
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def check_model_frozen(model):
    """
    Print each parameter's requires_grad status and the frozen ratio.
    """
    print("\nChecking model parameter status:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    frozen_params = sum(1 for param in model.parameters() if not param.requires_grad)
    total_params = sum(1 for param in model.parameters())
    print(f"\nFrozen parameters: {frozen_params}/{total_params} ({frozen_params/total_params*100:.2f}%)")
