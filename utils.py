import numpy as np
import torch as ch
from torchvision.models import *
from robustness.tools import helpers
from robustness.datasets import DATASETS
from robustness.tools.label_maps import CLASS_DICT
from robustness import model_utils, datasets
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

def load_model(arch, dataset=None):
    '''
    Load pretrained model with specified architecture.
    Args:
        arch (str): name of one of the pytorch pretrained models or 
                    "robust" for robust model
        dataset (dataset object): not None only for robust model
    Returns:
        model: loaded model
    '''
    
    if arch != 'robust':
        model = eval(arch)(pretrained=True).cuda()
        model.eval()
        pass
    else:
        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'resume_path': f'./models/RestrictedImageNet.pt'
        }

        model, _ = model_utils.make_and_restore_model(**model_kwargs)
        model.eval()
        try:
            model = model.module.model
        except:
            model = model.model
    return model

def load_dataset(dataset, batch_size, num_workers=1, data_path='./data'):
    '''
    Load pretrained model with specified architecture.
    Args:
        dataset (str): name of one of dataset 
                      ('restricted_imagenet' or 'imagenet')
        batch_size (int): batch size
        num_workers (int): number of workers
        data_path (str): path to data
    Returns:
        ds: dataset object
        loader: dataset loader
        norm: normalization function for dataset
        label_map: label map (class numbers to names) for dataset
    '''
    
    ds = DATASETS[dataset](data_path)
    loaders = ds.make_loaders(num_workers, batch_size, data_aug=False)
    normalization = helpers.InputNormalize(ds.mean, ds.std)
    label_map = CLASS_DICT['ImageNet'] if dataset == 'imagenet' else CLASS_DICT['RestrictedImageNet']
    return ds, loaders, normalization, label_map


def load_binary_dataset(batch_size, num_workers=1, classes=[0, 1], data_path='./data'):
    dataset, loaders, normalization, label_map = load_dataset('cifar',
                                                              batch_size=100,
                                                              num_workers=1)

    train_loader, val_loader = loaders
    
    def get_subset(loader, classes=[0, 1]):
        ims, targs = [], []
        for _, (im, targ) in enumerate(loader):
            for ci, c in enumerate(classes):
                idx = np.where(targ.numpy() == c)[0]
                if len(idx) == 0: continue
                ims.extend(im[idx])
                if ci == 0:
                    targs.extend(ch.zeros_like(targ[idx]))
                else:
                    targs.extend(ch.ones_like(targ[idx]))
        ims, targs = ch.stack(ims), ch.stack(targs, 0)
        idx  = np.arange(len(ims))
        np.random.shuffle(idx)
        return ims[idx], targs[idx]
   
    data = {}
    data['train'] = get_subset(train_loader, classes=classes)
    data['test'] = get_subset(val_loader, classes=classes)
    return data
        

def forward_pass(mod, im, normalization=None):
    '''
    Compute model output (logits) for a batch of inputs.
    Args:
        mod: model
        im (tensor): batch of images
        normalization (function): normalization function to be applied on inputs
        
    Returns:
        op: logits of model for given inputs
    '''
    if normalization is not None:
        im_norm = normalization(im)
    else:
        im_norm = im
    op = mod(im_norm.cuda())
    return op

def get_gradient(mod, im, targ, normalization, custom_loss=None):
    '''
    Compute model gradients w.r.t. inputs.
    Args:
        mod: model
        im (tensor): batch of images
        normalization (function): normalization function to be applied on inputs
        custom_loss (function): custom loss function to employ (optional)
        
    Returns:
        grad: model gradients w.r.t. inputs
        loss: model loss evaluated at inputs
    '''    
    def compute_loss(inp, target, normalization):
        if custom_loss is None:
            output = forward_pass(mod, inp, normalization)
            return ch.nn.CrossEntropyLoss()(output, target.cuda())
        else:
            return custom_loss(mod, inp, target.cuda(), normalization)
        
    x = im.clone().detach().requires_grad_(True)
    loss = compute_loss(x, targ, normalization)
    grad, = ch.autograd.grad(loss, [x])
    return grad.clone(), loss.detach().item()

def visualize_gradient(t):
    '''
    Visualize gradients of model. To transform gradient to image range [0, 1], we 
    subtract the mean, divide by 3 standard deviations, and then clip.
    
    Args:
        t (tensor): input tensor (usually gradients)
    '''  
    mt = ch.mean(t, dim=[2, 3], keepdim=True).expand_as(t)
    st = ch.std(t, dim=[2, 3], keepdim=True).expand_as(t)
    return ch.clamp((t - mt) / (3 * st) + 0.5, 0, 1) 

def L2PGD(mod, im, targ, normalization, step_size, Nsteps,
        eps=None, targeted=True, custom_loss=None):
    '''
    Compute L2 adversarial examples for given model.
    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        normalization (function): normalization function to be applied on inputs
        step_size (float): optimization step size
        Nsteps (int): number of optimization steps
        eps (float): radius of L2 ball
        targeted (bool): True if we want to maximize loss, else False
        custom_loss (function): custom loss function to employ (optional)
        
    Returns:
        x: batch of adversarial examples for input images
    '''      
    if custom_loss is None:
        loss_fn = ch.nn.CrossEntropyLoss()
    else:
        loss_fn = custom_loss 
        
    sign = -1 if targeted else 1
        
    it = tqdm(enumerate(range(Nsteps)), total=Nsteps)
    
    x = im.detach()
    l = len(x.shape) - 1
    
    for _, i in it:    
        x = x.clone().detach().requires_grad_(True)
        g, loss = get_gradient(mod, x, targ, normalization, 
                               custom_loss=custom_loss)
        
        it.set_description(f'Loss: {loss}')
        
        with ch.no_grad():
            
            # Compute gradient step 
            g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
            scaled_g = g / (g_norm + 1e-10)
            x += sign * scaled_g * step_size
            
            # Project back to L2 eps ball
            if eps is not None:
                diff = x - im
                diff = diff.renorm(p=2, dim=0, maxnorm=eps)
                x = im + diff
            x = ch.clamp(x, 0, 1)
    return x

def get_features(mod, im, normalization):
    '''
    Get feature representation of model  (output of layer before final linear 
    classifier) for given inputs.
    
    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        normalization (function): normalization function to be applied on inputs
        
    Returns:
        features: batch of features for input images
    '''   
    feature_rep = ch.nn.Sequential(*list(mod.children())[:-1])
    im_norm = normalization(im.cpu()).cuda()
    features = feature_rep(im_norm)[:, :, 0, 0]
    return features

## Helpers for training/evaluating linear classifiers
def accuracy(net, im, targ):
    '''
    Evaluate the accuracy of a given linear classifier.
    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        
    Returns:
        x: batch of adversarial examples for input images
    '''  
    op = net.forward(im).argmax(dim=1)
    acc = (op == targ).sum().item() / len(im) * 100
    return acc

class Linear(nn.Module):
    '''
    Class for linear classifiers.
    ''' 
    def __init__(self, Nfeatures, Nclasses):
        '''
        Initializes the linear classifier.
        Args:
            Nfeatures (int): Input dimension
            Nclasses (int): Number of classes
        ''' 
        super(Linear, self).__init__()
        self.fc = nn.Linear(Nfeatures, Nclasses)
    def forward(self, im):
        '''
        Perform a forward pass through the linear classifier.
        Args:
            im (tensor): batch of images

        Returns:
            pred (tensor): batch of logits
        ''' 
        imr = im.view(im.shape[0], -1)
        pred = self.fc(imr)
        return pred
    
def get_predictions(im, mod):
    '''
    Determine predictions of linear classifier.
    Args:
        im (tensor): batch of images
        mod: model

    Returns:
        op (tensor): batch of predicted labels
    ''' 
    with ch.no_grad():
        op = mod(im.cuda())
        op = op.argmax(dim=1)
    return op
    
def train_linear(data, 
                 Nclasses=2, 
                 step_size=0.1, 
                 iterations=1000,
                 log_iterations=500):
    '''
    Train a linear classifier on the input data.
    Args:
        data (dict): A dictionary containing train and test data
        Nclasses (int): Number of classes in the data
        step_size (float): Step size to use for gradient descent
        iterations (int): Number of steps to train the model for
        log_iterations (int): Frequency of printing/logging of accuracies

    Returns:
        store (dict): Train and eval logs
        final_net: trained linear classifier
    ''' 
    
    store = {'step': [], 'train': [], 'test': []}
    
    Nfeatures = int(np.prod(data['train'][0].shape[1:]))
    net = ch.nn.DataParallel(Linear(Nfeatures, Nclasses).cuda())
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=step_size)
    
    it = trange(iterations + 1)
    for k in it:
        if k % log_iterations == 0:
            store['step'].append(k)
            acc_log = []
            for name, (xs, ys) in data.items():
                xs, ys = xs.cuda(), ys.cuda()
                store[name].append(accuracy(net, xs, ys))
                acc_log.append(store[name][-1])
                if name == 'test' and len(store['test']) > 1 and \
                    store['test'][-1] > max(store['test'][:-1]):
                        params = [p.clone() for p in net.module.parameters()]
                   
        it.set_description(f"Train accuracy={acc_log[0]:.2f}, Test accuracy={acc_log[1]:.2f}")
        optimizer.zero_grad()
        xs, ys = data['train']
        xs, ys = xs.cuda(), ys.cuda()
        logits = net(xs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        
    final_net = Linear(data['train'][0].shape[1], Nclasses).cuda()    
    final_net.fc.weight.data = params[0]
    final_net.fc.weight.bias = params[1]
    final_net = ch.nn.DataParallel(final_net)
    
    xs, ys = data['test']
    xs, ys = xs.cuda(), ys.cuda()
    print(f"Final test accuracy: {accuracy(final_net, xs, ys):.2f}")
    return store, final_net