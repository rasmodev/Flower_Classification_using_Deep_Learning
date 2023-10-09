import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from model_functions import build_model, train_model, save_checkpoint
from utility_functions import load_data

def main():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save it as a checkpoint.")
    parser.add_argument('data_directory', metavar='data_directory', help='Path to the main data directory.')
    parser.add_argument('--arch', dest='arch', default='vgg16', help='Architecture (e.g., "vgg16", "resnet50").')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--save_dir', dest='save_dir', default='.', help='Directory to save checkpoint.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training if available.')

    args = parser.parse_args()

    # Load and preprocess data
    dataloaders, image_datasets = load_data(args.data_directory)

    # Build the model
    model = build_model(args.arch, args.hidden_units, len(image_datasets['train'].classes))

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Use GPU if available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, device, args.epochs)

    # Save the model checkpoint
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, args.epochs, optimizer, image_datasets)

if __name__ == '__main__':
    main()