from torchvision import transforms
from PIL import Image
import torch



def preprocess_image(image:Image.Image):

    image_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    image_tensor = image_data_transform(image).unsqueeze(0)
    return image_tensor

