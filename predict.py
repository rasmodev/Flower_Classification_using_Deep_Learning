import numpy as np
import argparse
import json
import torch
from PIL import Image
from model_functions import build_model
from utility_functions import load_data

def predict(image_path, checkpoint, top_k=5, category_names=None, gpu=False):

    # Load the mapping of categories to real names
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None

    # Load the model checkpoint
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    # Build the model
    model = build_model(arch, hidden_units, len(checkpoint['class_to_idx']))

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Use GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the image
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device, dtype=torch.float)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Calculate class probabilities and top classes
    probs, classes = torch.exp(output).topk(top_k)

    if cat_to_name:
        # Map class indices to class names
        idx_to_class = {val: key for key, val in checkpoint['class_to_idx'].items()}
        class_names = [cat_to_name[idx_to_class[idx]] for idx in classes[0].tolist()]
    else:
        class_names = [str(idx) for idx in classes[0].tolist()]

    return probs[0].tolist(), class_names

def process_image(image_path):
    # Open and preprocess an image
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    img = img.convert('RGB')
    img = torch.tensor(np.array(img) / 255.0).float() 
    img = (img - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    img = img.permute(2, 0, 1)
    return img

def main():
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained model.")
    parser.add_argument('image_path', metavar='image_path', help='Path to the input image.')
    parser.add_argument('checkpoint', metavar='checkpoint', help='Path to the model checkpoint.')
    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', dest='category_names', default=None, help='Path to a mapping of categories to real names.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for inference if available.')

    args = parser.parse_args()

    # Perform prediction
    probabilities, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

    # Print the results
    for prob, cls in zip(probabilities, classes):
        print(f"Class: {cls}, Probability: {prob:.4f}")

if __name__ == '__main__':
    main()