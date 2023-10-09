import torch
from torch import nn
from torch import optim
from torchvision import models

def build_model(arch, hidden_units, num_classes):
    """
    Build and return a pre-trained model with a custom classifier.

    Args:
        arch (str): Name of the pre-trained architecture (e.g., "vgg16", "resnet50").
        hidden_units (int): Number of hidden units in the classifier.
        num_classes (int): Number of output classes.

    Returns:
        model (nn.Module): The custom model with the specified architecture and classifier.
    """
    # Load the pre-trained model
    model = models.__dict__[arch](pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a custom classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    # Replace the pre-trained model's classifier with the custom classifier
    model.classifier = classifier

    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs=10):
    """
    Train a model on a dataset.

    Args:
        model (nn.Module): The model to be trained.
        dataloaders (dict): A dictionary containing DataLoader objects for train, validation, and test data.
        criterion: The loss function.
        optimizer: The optimizer for updating model weights.
        device: The device to use for training (e.g., "cuda" or "cpu").
        epochs (int): Number of training epochs.

    Returns:
        model (nn.Module): The trained model.
    """
    # Move model to the specified device
    model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        validation_loss = 0.0
        accuracy = 0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        # Calculate the average training and validation loss for this epoch
        avg_train_loss = running_loss / len(dataloaders['train'])
        avg_valid_loss = validation_loss / len(dataloaders['valid'])
        
        # Calculate the average validation accuracy for this epoch
        avg_valid_accuracy = accuracy / len(dataloaders['valid'])
        
        # Print training and validation statistics for this epoch
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {avg_train_loss:.2f}.. "
              f"Validation loss: {avg_valid_loss:.2f}.. "
              f"Validation accuracy: {avg_valid_accuracy:.2f}")
    
    return model


def save_checkpoint(model, save_dir, arch, hidden_units, epochs, optimizer, image_datasets):
    """
    Save the model checkpoint to a file.

    Args:
        model (nn.Module): The trained model.
        save_dir (str): Directory to save the checkpoint file.
        arch (str): Name of the architecture used.
        hidden_units (int): Number of hidden units in the classifier.
        epochs (int): Number of training epochs.
        optimizer: The optimizer used for training.
        image_datasets (dict): A dictionary containing datasets for train, validation, and test data.
    """
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'model_state_dict': model.state_dict()
    }
    
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")