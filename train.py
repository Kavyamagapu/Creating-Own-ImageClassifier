import argparse
import time
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json

from torch import nn,optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def validation():
    print("Parameters Validating")
    if(args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled..but no GPU detected")
    if(not os.path.isdir(args.data_directory)):
        raise Exception("directory does not exist!")
    data_dir = os.listdir(args.data_directory)
    if(not set(data_dir).issubset({'test','train','valid'})):
        raise Exception("Missing: test, train or valid sub-directories")
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet')

def process_data(data_dir):
    print("Processing data into iterations")
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
            
    data_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                    ])
    modified_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                        ])
     
    image_datasets = datasets.ImageFolder(data_dir, transform = modified_transforms)
    image_datasets_train = datasets.ImageFolder(train_dir, transform = modified_transforms)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform = modified_transforms)
    image_datasets_test = datasets.ImageFolder(test_dir, transform = modified_transforms)
                        
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = 32, shuffle=True)
    trainloaders = torch.utils.data.DataLoader(image_datasets_train,batch_size=32,shuffle=True)
    validloaders = torch.utils.data.DataLoader(image_datasets_valid,batch_size=32,shuffle=True)
    testloaders = torch.utils.data.DataLoader(image_datasets_test,batch_size=32,shuffle=True)
                        
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    loaders = {'train':trainloaders, 'valid':validloaders, 'test':testloaders, 'labels':cat_to_name}
    return loaders
                        
    
def get_data():
    print("Retreiving Data")
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    data_dir = [train_dir,test_dir,valid_dir]
    return process_data(data_dir)
                        
def build_model(data):
    print("building model object")
    if(args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = args.arch
    if(arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node = 25088
    elif (arch_type == 'densenet'):
         model = models.densenet121(pretrained-True)
         input_node = 1024
    if (args.hidden_units is None):
        hidden_units = 2048
    else:
        hidden_units = args.hidden_units
    for param in model.parameters():
         param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2048)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(2048, 256)),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    return model
                  
def test_accuracy(model,loader,device = "cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images);
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0);
            correct += (predicted == labels).sum().item()
    return correct / total
def train(model,data):
    print("training model")
    
    print_every = 40
    
    if(args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    if(args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
    if(args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learn_rate = float(learn_rate)
    epochs = int(epochs)
             
    trainloader = data['train']
    validloader = data['valid']
    testloader = data['test']
                        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
                       
    steps = 0
    model.to(device)
                        
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            print("--")            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                        
            running_loss += loss.item()
                        
            if steps % print_every == 0:
                valid_accuracy = test_accuracy(model, validloader,device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))
                running_loss = 0
    print("DONE TRAINNG")
    test_result = test_accuracy(model, testloader, device)
    print('final accuracy on test set: {}'.format(test_result))
    return model
                        
def save_model(model):
    print("saving model")
    if(args.save_dir is None):
        save_dir = 'check.pth'
    else:
        save_dir = args.save_dir
    checkpoint = {
                   'model' : model.cpu(),
                   'features' : model.features,
                   'classifier' : model.classifier,
                   'state_dict' : model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0
                    
def create_model():
    validation()
    data = get_data()
    model = build_model(data)
    model = train(model,data)
    save_model(model)
    return None
                        
def parse():
    parser = argparse.ArgumentParser(description = 'Train a neural network with open of many options!')
    parser.add_argument('data_directory', help = 'data directory (required)')
    parser.add_argument('--save_dir', help = 'directory to save a neural network')     
    parser.add_argument('--arch', help = 'models to use options')
    parser.add_argument('--learning_rate', help = 'learning_rate')
    parser.add_argument('--hidden_units', help = 'number of hidden units')
    parser.add_argument('--epochs', help = 'epochs')
    parser.add_argument('--gpu', action = 'store_true', help = 'gpu')
    args = parser.parse_args()
    return args
                        
                        
def main():
    print("Creating Deep Learning Model")
    global args
    args = parse()
    create_model()
    print("Model Ready")
    return None
                        
main()
   
                        
