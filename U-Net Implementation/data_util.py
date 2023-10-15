import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    '''
    Saves the model checkpoint.
    '''
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    '''
    Loads the model checkpoint.
    '''
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
