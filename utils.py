import torch

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def load_model(model_class, name, device=None):
    if device is None:
        device = get_device()
        
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=get_device()))
    model.to(get_device())
    return model