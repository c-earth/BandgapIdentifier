import torch
import torch.nn as nn
import torch.nn.functional as F
import random

dtype = torch.float64

class BandAugmentations():
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, bands):
        pivot = [list(range(len(bands))),[random.randint(0, bands.shape[-1]-1)for i in range(len(bands))]]
        inc_aug = bands + torch.abs(torch.normal(torch.zeros(bands.shape), self.delta*torch.ones(bands.shape)))
        dec_aug = bands - torch.abs(torch.normal(torch.zeros(bands.shape), self.delta*torch.ones(bands.shape)))
        inc_aug[pivot] = bands[pivot]
        dec_aug[pivot] = bands[pivot]
        return torch.concat([inc_aug, dec_aug], dim = 0)

class NCELoss(torch.nn.Module):
    def __init__(self, sim_fn, T = 1):
        super(NCELoss, self).__init__()
        self.sim_fn = sim_fn
        self.T = T

    def forward(self, features, not_neg_mask, pos_mask):
        sims = torch.masked_fill(self.sim_fn(features) / self.T, not_neg_mask, -1e10)
        return torch.mean(-sims[pos_mask] + torch.logsumexp(sims, dim = -1))
    
def conv3x3x3(in_channels, out_channels, stride = 1, groups = 1, dilation = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = dilation, groups = groups, bias = False, dilation = dilation)


def conv1x1x1(in_channels, out_channels, stride = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)

class Block(nn.Module):
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
    def __init__(self, layers, num_classes = 1000, zero_init_residual = False, groups =1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
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
        self.conv1 = nn.Conv3d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(Block, 64, layers[0])
        self.layer2 = self._make_layer(Block, 128, layers[1], stride = 2, dilate = replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Block, 256, layers[2], stride = 2, dilate = replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(Block, 512, layers[3], stride = 2, dilate = replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)

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
        return self._forward_impl(bands)

class ProjectionNetwork(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.mlp1 = nn.Linear(n_features, n_features)
        self.mlp2 = nn.Linear(n_features, n_features)
    
    def forward(self, features):
        x = F.relu(self.mlp1(features))
        x = self.mlp2(x)
        return x

class ContrastiveNetwork(torch.nn.Module):
    def __init__(self, feature_model, projection_model):
        super().__init__()
        self.feature_model = feature_model
        self.projection_model = projection_model
    
    def forward(self, data):
        features = self.feature_model(data.bands)
        features = self.projection_model(features)
        return features

def cos_sim(features):
    nfeatures = torch.nn.functional.normalize(features, p = 2, dim = 1)
    return torch.matmul(nfeatures, nfeatures.T)

from torch_geometric.loader import DataLoader
import time
from matplotlib.pyplot import plt
import numpy as np
import math

def train(feature_model,
          projection_model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          batch_size,
          k_fold):
    model = CantrastiveNetwork(feature_model, projection_model)
    model.to(device)
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

    tr_sets = torch.utils.data.random_split(tr_set, tr_nums)
    te_loader = DataLoader(te_set, batch_size = batch_size)
    for step in range(max_iter):
        k = step % k_fold 
        tr_loader = DataLoader(torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), batch_size = batch_size, shuffle=True)
        va_loader = DataLoader(tr_sets[k], batch_size = batch_size)
        model.train()
        N = len(tr_loader)
        for i, d in enumerate(tr_loader):
            start = time.time()
            d.to(device)
            features = model(d)
            loss = loss_fn(batch, not_neg_mask, pos_mask).cpu()  #! define not_neg_mask, pos_mask
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end = '\r')

        end_time = time.time()
        wall = end_time - start_time
        print(wall)
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, va_loader, loss_fn, device)
            train_avg_loss = evaluate(model, tr_loader, loss_fn, device)
            history.append({
                            'step': s0 + step,
                            'wall': wall,
                            'batch': {
                                    'loss': loss.item(),
                                    },
                            'valid': {
                                    'loss': valid_avg_loss,
                                    },
                            'train': {
                                    'loss': train_avg_loss,
                                    },
                           })
            results = {
                        'history': history,
                        'state': model.state_dict()
                      }
            results_feature = {
                        'state': model.feature_model.state_dict()
                      }
            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss:8.20f}   " +
                  f"valid loss = {valid_avg_loss:8.20f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(run_name + '_contrastive.torch', 'wb') as f:
                torch.save(results, f)
            with open(run_name + '_feature.torch', 'wb') as f:
                torch.save(results_feature, f)

            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
            record_lines.append(record_line)

            loss_plot(run_name, device, './models/' + run_name)
            loss_test_plot(model, device, './models/' + run_name, te_loader, loss_fn)

            # plot the output by the model.feature_model
            
        text_file = open(run_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()

        if scheduler is not None:
            scheduler.step()

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))
            
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            features = model(d)
            loss = loss_fn(batch, not_neg_mask, pos_mask).cpu()  #! define not_neg_mask, pos_mask
            loss_cumulative += loss.detach().item()
    return loss_cumulative/len(dataloader)


def loss_plot(model_file, device, fig_file):
    history = torch.load(model_file + '.torch', map_location = device)['history']
    steps = [d['step'] + 1 for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, loss_train, 'o-', label='Training')
    ax.plot(steps, loss_valid, 'o-', label='Validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file  + '_loss_train_valid.png')
    plt.close()

def loss_test_plot(model, device, fig_file, dataloader, loss_fn):
    loss_test = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            batch = model(d)
            loss = loss_fn(batch, not_neg_mask, pos_mask).cpu()  #! define not_neg_mask, pos_mask
            loss_test.append(loss)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(loss_test), label = 'testing loss: ' + str(np.mean(loss_test)))
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file + '_loss_test.png')
    plt.close()
