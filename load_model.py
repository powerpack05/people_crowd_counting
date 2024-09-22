import torch
from model import CSRNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_the_model(weights_path="first.model.path"):

    model = CSRNet().to(device)
    model.load_state_dict(torch.load(weights_path,map_location=device))
    model.eval()
    return model

if __name__ == '__main__':

    a = load_the_model(r'models/first_model.pth')
    print(a)