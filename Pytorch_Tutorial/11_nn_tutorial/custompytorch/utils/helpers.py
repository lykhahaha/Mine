import torch
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import re

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, num_epochs=25):
    """
    Scheduling the learning rate
    Saving the best model
    Arguments:
    model: nn Modules
    dataloaders: {'train': torch.utils.data.DataLoader, 'val': torch.utils.data.DataLoader}
    dataset_sizes: {'train': dataset_sizes of train, 'val': dataset_sizes of test}
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.

    for e in range(num_epochs):
        start = time.time()

        statistics = {
            'train': {
                'loss': 0.,
                'acc': 0.
            },
            'val': {
                'loss':0.,
                'acc': 0.
            }
        }

        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode

            # loop over dataloader
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero out parameter gradients
                optimizer.zero_grad()

                # Forward pass, track history in train phase
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1) # torch.max return 2 tensors: first is max value, second is argmax value
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                statistics[phase]['loss'] += loss.item() * inputs.size(0)
                statistics[phase]['acc'] += (preds == labels.data).sum().item()

            statistics[phase] = {key: statistics[phase][key]/dataset_sizes[phase] for key in statistics[phase].keys()}
        
        time_elapsed = time.time() - start
        print(f"[INFO]Epoch {e+1}/{num_epochs} - {time_elapsed:.2f}s - Loss: {statistics['train']['loss']:.5f}, Accuracy: {statistics['train']['acc']:.5f}, Validation loss: {statistics['val']['loss']:.5f}, Validation accuracy: {statistics['val']['acc']:.5f}")
        
        if best_val_acc < statistics['val']['acc']:
            best_val_acc = statistics['val']['acc']
            best_model_wts = copy.deepcopy(model.state_dict())

    # load best weights
    model.load_state_dict(best_model_wts)
    return model

def imshow(inp, title=None):
    """
    Imshow for Tensor
    """
    inp = inp.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)

def visualize_model(model, dataloaders, class_names, file_names=None, num_images=6):
    """
    Generic function to display predictions for a few images
    Arguments:
    class_names: ['ant', 'bee']
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    model.eval()
    fig = plt.figure()
    image_num = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            for j in range(inputs.size(0)):
                image_num += 1

                if j == num_images:
                    if file_names:
                        fig.savefig(file_names)
                        plt.close(fig)
                    else:
                        plt.imshow(fig)
                    model.train()
                    return

                ax = plt.subplot(num_images//2, 2, image_num)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

# Lowercase and remove non-letter characters
def normalize_string(str):
    str = str.lower()
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^a-zA-Z.!?]+", r" ", str)
    return str

# Takes string sentence and returns sentence of word indexes
def indexes_from_sentence(voc, sentence, EOS_TOKEN = 2):
    return [voc.word2index[word] for word in sentence.split()] + [EOS_TOKEN]