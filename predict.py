import argparse
import json
import PIL
from PIL import Image
import torch
import numpy as np
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Point to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Point to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.', default=3)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg16(pretrained=True)
    for param in model.parameters(): 
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image) / 255.0  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))  
    return np_image

def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        category_names = json.load(f)
    return category_names

def predict(image_path, model, top_k=3, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()  
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)  # Add batch dimension

    if gpu and torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()

    with torch.no_grad():
        output = model(image)
    
    probabilities = torch.exp(output)
    top_probs, top_classes = probabilities.topk(top_k)

    return top_probs.cpu().numpy().flatten(), top_classes.cpu().numpy().flatten()

if __name__ == "__main__":
    args = arg_parser()
    model = load_checkpoint(args.checkpoint)

 
    category_names = load_category_names(args.category_names)

    top_probs, top_classes = predict(args.image, model, args.top_k, args.gpu)

    
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}  

    top_class_names = [category_names[idx_to_class[str(cls)]] for cls in top_classes]

    print("Top K Probabilities:", top_probs)
    print("Top K Classes:", top_classes)
    print("Top K Class Names:", top_class_names)
