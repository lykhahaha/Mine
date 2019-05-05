import torch
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import re
from torch import nn
from torchvision import models
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, num_epochs=25, is_inception=False):
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
                    # Special case for inception b/c in training, it has an auxiliary output. In train mode, we calculate the loss by summing final output and auxiliary output, but in val/test mode, we onl;y consider final output
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss_1 = criterion(outputs, labels)
                        loss_2 = criterion(aux_outputs, labels)
                        loss = loss_1 + 0.4 * loss_2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1) # torch.max return 2 tensors: first is max value, second is argmax value

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

def initialize_model_finetuning(model_name, num_classes, use_pretrained=True):
    if model_name not in ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']:
        raise ValueError(model_name, ' is not in valid model name')
    
    model = None
    input_size = 224

    if model_name == 'resnet':
        model = models.resnet18(pretrained=use_pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'vgg':
        model = models.vgg11_bn(pretrained=use_pretrained)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'squeezenet':
        model = models.squeezenet1_0(pretrained=use_pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=use_pretrained)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'inception':
        model = models.inception_v3(pretrained==use_pretrained)
        for param in model.parameters():
            param.requires_grad = False
        
        # Handle auxiliary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    
    return model, input_size

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

def convert_torch2onnx(model, model_url, onnx_file, batch_size=1, size=224):
    map_location = lambda storage, loc: storage
    model.load_state_dict(torch.utils.model_zoo.load_url(model_url, map_location=map_location))
    model.eval()

    # torch.onnx._export(): execute the model, record a trace of what operators are used to compute outputs
    # B/c _export runs model, we need to provide input tensor x
    # Value of tensor x is not important, it can be an image or a random tensor as long as right size

    # Define input tensor
    inputs = torch.randn(batch_size, 1, size, size, requires_grad=True)

    # Export model
    # Normally torch_out is not necessary, but here we use it to verify that model we exported has same values when running in Caffe2
    torch_out = torch.onnx._export(model, inputs, onnx_file, export_params=True) # export_params: store trained parameter weights inside model file

    return torch_out, inputs

def convert_onnx2caffe2(torch_out, inputs, onnx_file):
    # Load ONNX ModelProto object. Model is a standard Python protobuf object
    print('[INFO] loading onnx model...')
    model = onnx.load(onnx_file)

    # Prepare caffe2 backend for executing model
    prepare_backend = onnx_caffe2_backend.prepare(model)
    # Construct a map from input names to Tensor data
    W = {model.graph.input[0].name: inputs.data.numpy()}
    # Convert ONNX model into a Caffe2 NetDef that can execute it
    caffe2_out = prepare_backend.run(W)[0]
    # Verify numerical correctness upto 3 decimal places
    np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), caffe2_out, decimal=3)
    print('[INFO] Conversion is done')
    return prepare_backend

# Lowercase and remove non-letter characters
def normalize_string(str):
    str = str.lower()
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^a-zA-Z.!?]+", r" ", str)
    return str

# Takes string sentence and returns sentence of word indexes
def indexes_from_sentence(voc, sentence, EOS_TOKEN = 2):
    return [voc.word2index[word] for word in sentence.split()] + [EOS_TOKEN]