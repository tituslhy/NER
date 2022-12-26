import torch

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def load_model(name, model_class, device = None):
    """Loads a saved model for evaluation or inference

    Args:
        name (_type_): _description_
        model_class (_type_, optional): _description_. Defaults to None.
        jit (bool, optional): _description_. Defaults to False.
        device (_type_, optional): _description_. Defaults to None.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if device is None:
        device = get_device()
    
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=get_device()))
    model.to(get_device())
            
    return model