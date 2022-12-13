import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

class BandAugmentations():
    '''
    phonon frequency response augmentation by fixing one point of each band to remain unchanged 
    after augmentation. the augmented bands satisfy the phonon band criteria.
    '''
    def __init__(self, fun = (lambda x : 1/x)):
        '''
        arguments:
            fun: fourier component decaying function
        '''
        self.fun = fun

    def __call__(self, bands):
        '''
        arguments:
            bands: phonon frequency responses data object
        return:
            augmented phonon frequency responses
        '''
        dim_n, dim_ch, dim_a, dim_b, dim_c = bands.shape

        # set the sampling distribution for fourier components
        sigma_a, sigma_b, sigma_c = self.fun(np.arange(dim_a) + 1), self.fun(np.arange(dim_b) + 1), self.fun(np.arange(dim_c) + 1)

        # get fourier components for each axis
        f_sin_a, f_cos_a = np.random.normal(scale = sigma_a, size = (dim_n, dim_ch, dim_a)), np.random.normal(scale = sigma_a, size = (dim_n, dim_ch, dim_a))
        f_sin_b, f_cos_b = np.random.normal(scale = sigma_b, size = (dim_n, dim_ch, dim_b)), np.random.normal(scale = sigma_b, size = (dim_n, dim_ch, dim_b))
        f_sin_c, f_cos_c = np.random.normal(scale = sigma_c, size = (dim_n, dim_ch, dim_c)), np.random.normal(scale = sigma_c, size = (dim_n, dim_ch, dim_c))
        
        # select points of each band to remain unchanged
        pivot_a, pivot_b, pivot_c = np.random.randint(dim_a, size = (dim_n, dim_ch, 1)), np.random.randint(dim_b, size = (dim_n, dim_ch, 1)), np.random.randint(dim_c, size = (dim_n, dim_ch, 1))
        diff_a, diff_b, diff_c = np.zeros((dim_n, dim_ch, dim_a)), np.zeros((dim_n, dim_ch, dim_b)), np.zeros((dim_n, dim_ch, dim_c))
        
        # add all fourier components and add on to the input to get augmented version
        idx, idy, idz = (np.arange(dim_a) - pivot_a) % dim_a, (np.arange(dim_b) - pivot_b) % dim_b, (np.arange(dim_c) - pivot_c) % dim_c
        for i in range(dim_a):
            diff_a += f_sin_a[:, :, i:i+1] * np.sin(2*np.pi*(i+1)*idx/dim_a) + f_cos_a[:, :, i:i+1] * np.cos(2*np.pi*(i+1)*idx/dim_a)
        for i in range(dim_b):
            diff_b += f_sin_b[:, :, i:i+1] * np.sin(2*np.pi*(i+1)*idy/dim_b) + f_cos_b[:, :, i:i+1] * np.cos(2*np.pi*(i+1)*idy/dim_b)
        for i in range(dim_c):
            diff_c += f_sin_c[:, :, i:i+1] * np.sin(2*np.pi*(i+1)*idz/dim_c) + f_cos_c[:, :, i:i+1] * np.cos(2*np.pi*(i+1)*idz/dim_c)
        diff = diff_a.reshape(dim_n, dim_ch, -1, 1, 1) + diff_b.reshape(dim_n, dim_ch, 1, -1, 1) + diff_c.reshape(dim_n, dim_ch, 1, 1, -1)
        index = np.array(list(np.ndindex(dim_n, dim_ch))).T
        return  bands + np.std(bands.reshape((dim_n, -1)), axis = -1).reshape((dim_n, 1, 1, 1, 1))*np.abs(diff - diff[index[0], index[1], pivot_a.flatten(), pivot_b.flatten(), pivot_c.flatten()].reshape(dim_n, dim_ch, 1, 1, 1)) * np.random.choice([-1, 1], 1)

class NCELoss(torch.nn.Module):
    '''
    noise contrastive estimation (NCE) loss function
    '''
    def __init__(self, sim_fn, T = 1):
        '''
        arguments:
            sim_fn: similarity function
            T: exponential weight factor
        '''
        super().__init__()
        self.sim_fn = sim_fn
        self.T = T

    def forward(self, features, non_neg_mask, pos_mask):
        '''
        arguments:
            features: feature vectors on the projected space
            non_neg_mask: mask for non-negative sample pairs of bands 
                          with respect to the original material's bands
            pos_mask: mask for positive sample pairs of bands
        return:
            NCE loss value
        '''
        sims = torch.masked_fill(self.sim_fn(features) / self.T, non_neg_mask, -1e15)
        return torch.mean(-sims[pos_mask][:len(features)//2] + torch.logsumexp(sims, dim = -1)[:len(features)//2])
    
def conv3x3x3(in_channels, out_channels, stride = 1, groups = 1, dilation = 1):
    '''
    modified 3D convolution (3, 3, 3) kernel from ResNet
    '''
    return nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = dilation, groups = groups, bias = False, dilation = dilation)


def conv1x1x1(in_channels, out_channels, stride = 1):
    '''
    modified 3D convolution (1, 1, 1) kernel from ResNet
    '''
    return nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)

class Block(nn.Module):
    '''
    modified 3D block layer from ResNet
    '''
    expansion = 4

    def __init__(self, in_channels, channels, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(channels * (base_width / 64.0)) * groups
        self.conv1 = conv1x1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FeatureNetwork(torch.nn.Module):
    '''
    model for deep neural network that encode each phonon band into a feature vector.
    This model modifies ResNet, deep residual learning network, such that it can handle
    3D images as input. 
    '''
    def __init__(self, n_layers, n_features, zero_init_residual = False, groups =1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        '''
        arguments:
            n_layers: ResNet hyper-parameters: '[2, 2, 2, 2]' is for ResNet18
            n_features: feature vector space dimensions
        '''
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                            f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(64, n_layers[0])
        self.layer2 = self._make_layer(128, n_layers[1], stride = 2, dilate = replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, n_layers[2], stride = 2, dilate = replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, n_layers[3], stride = 2, dilate = replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * Block.expansion, n_features)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Block) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != planes * Block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_channels, planes * Block.expansion, stride),
                norm_layer(planes * Block.expansion),
            )

        layers = []
        layers.append(
            Block(
                self.in_channels, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.in_channels = planes * Block.expansion
        for _ in range(1, blocks):
            layers.append(
                Block(
                    self.in_channels,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, bands):
        '''
        arguments:
            bands: phonon frequency responses data object
        return:
            feature vector of each phonon band
        '''
        return self._forward_impl(bands)

class ProjectionNetwork(torch.nn.Module):
    '''
    model for projecting feature vectors  into a space used
    for measuring noise contrastive estimation (NCE) loss.
    '''
    def __init__(self, n_features):
        '''
        arguments:
            n_features: feature vector space dimensions
        '''
        super().__init__()
        self.mlp1 = nn.Linear(n_features, n_features)
        self.mlp2 = nn.Linear(n_features, n_features)
    
    def forward(self, features):
        '''
        arguments:
            features: feature vectors of each phonon band from feature encoder
        return:
            feature vector of each phonon band on the projected space
        '''
        x = F.relu(self.mlp1(features))
        x = self.mlp2(x)
        return x

class ContrastiveNetwork(torch.nn.Module):
    '''
    model for contrastive learning. Attach projection network to the feature network.
    '''
    def __init__(self, feature_model, projection_model):
        '''
        arguments:
            feature_model: FeatureNetwork model
            projection_model: ProjectionNetwork model
        '''
        super().__init__()
        self.feature_model = feature_model
        self.projection_model = projection_model
    
    def forward(self, phn):
        '''
        arguments:
            phn: phonon frequency responses data object
        return:
            feature vector of each phonon band on the projected space
        '''
        features = self.feature_model(phn)
        features = self.projection_model(features)
        return features

def cos_sim(features):
    '''
    calculate cos similarity of every pair of features
    arguments:
        features: feature vectors for all bands of the material
    return:
        cos similarity matrix
    '''
    nfeatures = torch.nn.functional.normalize(features, p = 2, dim = 1)
    return torch.matmul(nfeatures, nfeatures.T)

def loglinspace(rate, step, end = None):
    '''
    calculate the time steps for model evaluation such that 
    each subsequence time steps are exponentially further apart
    arguments:
        rate: evaluation rate
        step: time step
        end: end step
    return:
        next time step for evaluating the model
    '''
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))
            
def evaluate(model, dataloader, loss_fn, device, BA):
    '''
    evaluate a model on data in dataloader
    arguments:
        model: contrastive learning model
        dataloader: torch dataloader
        loss_fn: loss function
        device: torch device (cpu/cuda)
        BA: band augmentation function
    return:
        average loss of the data in dataloader
    '''
    model.eval()
    loss_cumulative = 0.

    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            
            # augment positive pair and get feature of all bands
            phns = torch.concat((d.phns, torch.from_numpy(BA(d.phns.cpu().numpy())).to(device)), dim = 0)
            features = model(phns)

            # loss calculation and backward propagation
            loss = loss_fn(features, d.non_neg_mask, d.pos_mask).cpu()
            loss_cumulative += loss.detach().item()

    return loss_cumulative/len(dataloader)      

def loss_plot(run_name, model_dir, device):
    '''
    plot loss evolution
    arguments:
        run_name: model name
        model_dir: folder for output
        device: torch device (cpu/cuda)
    return:
        None
    '''
    history = torch.load(f'{model_dir}{run_name}_contrastive.torch', map_location = device)['history']
    steps = [d['step'] + 1 for d in history]
    tr_loss = [d['train']['loss'] for d in history]
    va_loss = [d['valid']['loss'] for d in history]
    te_loss = [d['test']['loss'] for d in history]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, tr_loss, 'o-', label = 'Training')
    ax.plot(steps, va_loss, 'o-', label = 'Validation')
    ax.plot(steps, te_loss, 'o-', label = 'Test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(f'{model_dir}{run_name}_loss_train_valid.png')
    plt.close()

def train(model, opt, tr_sets, te_set, loss_fn, run_name, model_dir, max_iter, scheduler, device, batch_size, k_fold):
    '''
    main training function
    arguments:
        model: contrastive learning model
        opt: torch optimizer
        tr_sets: training set
        te_set: testing set
        loss_fn: loss function
        run_name: model name
        model_dir: folder for output
        max_iter: total epochs
        scheduler: learning rate decay scheduler
        device: torch device (cpu/cuda)
        batch_size: always set to 1 to go material by material
        k_fold: number of cross-validation blocks
    return:
        None
    '''
    # set up model
    model.to(device)

    # set up model tracking
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    record_lines = []
    try:
        print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        print('There is no existing model')
        results = {}
        history = []
        s0 = 0
    else:
        print('Use torch.load to load the existing model: ' + run_name + '.torch')
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1

    # setup phonon band augmentation
    BA = BandAugmentations()

    for step in range(max_iter):
        
        # merge cross-validation blocks for each step, and prepare data loader for training, validating, and testing
        k = step % k_fold 
        tr_loader = DataLoader(torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), batch_size = batch_size, shuffle=True)
        va_loader = DataLoader(tr_sets[k], batch_size = batch_size)
        te_loader = DataLoader(te_set, batch_size = batch_size)

        # train model
        model.train()
        N = len(tr_loader)
        for i, d in enumerate(tr_loader):
            start = time.time()
            d.to(device)

            # augment positive pair and get feature of all bands
            phns = torch.concat((d.phns, torch.from_numpy(BA(d.phns.cpu().numpy())).to(device)), dim = 0)
            features = model(phns)

            # loss calculation and backward propagation
            loss = loss_fn(features, d.non_neg_mask, d.pos_mask).cpu()
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end = '\r')
        wall = time.time() - start_time

        # keep track record of trained model
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            tr_avg_loss = evaluate(model, tr_loader, loss_fn, device, BA)
            va_avg_loss = evaluate(model, va_loader, loss_fn, device, BA)
            te_avg_loss = evaluate(model, te_loader, loss_fn, device, BA)
            history.append({
                            'step': s0 + step,
                            'wall': wall,
                            'batch': {
                                    'loss': loss.item(),
                                    },
                            'train': {
                                    'loss': tr_avg_loss,
                                    },
                            'valid': {
                                    'loss': va_avg_loss,
                                    },
                            'test': {
                                    'loss': te_avg_loss,
                                    },
                           })
            results = {
                        'history': history,
                        'state': model.state_dict()
                      }
            results_feature = {
                        'state': model.feature_model.state_dict()
                      }
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(wall))
            print(f'Iteration {step+1:4d}   ' +
                  f'train loss = {tr_avg_loss:8.20f}   ' +
                  f'valid loss = {va_avg_loss:8.20f}   ' +
                  f'test loss = {te_avg_loss:8.20f}   ' +
                  f'elapsed time = {elapsed_time}')

            # save models
            with open(f'{model_dir}{run_name}_contrastive.torch', 'wb') as f:
                torch.save(results, f)
            with open(f'{model_dir}{run_name}_feature.torch', 'wb') as f:
                torch.save(results_feature, f)

            record_line = '%d\t%.20f\t%.20f\t%.20f'%(step, tr_avg_loss, va_avg_loss, te_avg_loss)
            record_lines.append(record_line)

            # plot loss evolution
            loss_plot(run_name, model_dir, device)
            
        # save loss evolution
        text_file = open(f'{model_dir}{run_name}.txt', 'w')
        for line in record_lines:
            text_file.write(line + '\n')
        text_file.close()

        if scheduler is not None:
            scheduler.step()
