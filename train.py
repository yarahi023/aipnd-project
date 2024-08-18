import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
from os.path import isdir
from torch.utils.data import DataLoader

def args_parser():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str, help='Deep NN architecture, options: "vgg16" and "vgg13"')
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.0001, help='learning_rate options: 0.0001 and 0.01')
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096, help='hidden units options: 4096 and 512')
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=4, help='epochs options:4 and 20')
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu",  help='options:GPU and cpU') 
    args = parser.parse_args()
    
    if args.learning_rate not in [0.0001, 0.01]:
        print("Invalid learning rate specified. Setting to default: 0.0001")
        args.learning_rate = 0.0001
    
    return args

def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                           transforms.RandomRotation(30),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data
 
def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(size=256),
                                          transforms.CenterCrop(size=224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
                                          
def data_loader(image_datasets, train=True):
    if train:  
        loader = DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    else: 
        loader = DataLoader(image_datasets['valid'], batch_size=64, shuffle=False)
                                          
    return loader  
                                          
def check_gpu(gpu_arg):
    if gpu_arg and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("CUDA was not found on device, using CPU instead.")
        return torch.device("cpu")
                             
def primaryloader_model(architecture="vgg16"):
          model = models.vgg16(pretrained=True)
          model.name = "vgg16"
        
     for param in model.features.parameters():
         param.requires_grad = False
     return model

def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
    ('inputs', nn.Linear(25088, 4096)),
    ('batchnorm1', nn.BatchNorm1d(4096)),
    ('dropout', nn.Dropout(p=0.5)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(4096, 102)),
    ('batchnorm2', nn.BatchNorm1d(102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
                                          
    for param in model.classifier.parameters():
        param.requires_grad = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
                                          
model.classifier = classifier
    return classifier                                          

def evaluate(model, dataloader, criterion, device):
    model.eval() 
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            ps = torch.exp(outputs)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            correct += equals.sum().item()
            total += labels.size(0)

    average_loss = test_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy

def train_model(model, dataloaders, criterion, optimizer, device, epochs=4, print_every=50):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        model.train() 
        running_loss = 0
        steps = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if steps % print_every == 0:
                train_loss = running_loss / print_every
                test_loss, test_accuracy = evaluate(model, dataloaders['test'], criterion, device)

                print(f"Epoch {epoch + 1}/{epochs}, steps: {steps}, "
                      f"Train loss: {train_loss:.4f}, "
                      f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

                running_loss = 0

        validation_loss, validation_accuracy = evaluate(model, dataloaders['validation'], criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Validation loss: {validation_loss:.4f}, "
              f"Validation accuracy: {validation_accuracy:.4f}")

def calculate_accuracy(model, dataloaders, device):
    correct = total = 0

    with torch.no_grad():
        model.eval()
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)
            predicted = outputs.argmax(dim=1)  
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy
                                          
def initial_checkpoint(model, save_dir, train_data, optimizer):
    if save_dir is None:
        print("Checkpoint directory not specified and not saved.")
        return

    if not isdir(save_dir):
        print("Directory not found and not saved.")
        return

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'hidden_layer1': 4096,
        'dropout': 0.5,  
        'epochs': 4
}
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')


def main():
    args = args_parser()

    data_dir = 'flowers'
    train_dir, valid_dir, test_dir = [f"{data_dir}/{x}" for x in ['train', 'valid', 'test']]
    
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
   
    datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    
    dataloaders = {
        'train': data_loader({'train': train_data}),
        'validation': data_loader({'validation': valid_data}, train=False),
        'test': data_loader({'test': test_data}, train=False)
    }
    
    model = primaryloader_model(architecture=args.arch)
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)

    device = check_gpu(gpu_arg=args.gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    train_model(model, dataloaders, criterion, optimizer, device, args.epochs, print_every=50)

    print("\nTraining process done.")
    
    calculate_accuracy(model, dataloaders['test'], device)

    initial_checkpoint(model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
           
