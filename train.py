import time
from pathlib import Path
import os
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
from scipy.io import savemat
import shutil
import torch.optim as optim
from imageio import imread
import glob
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import auc
import pickle
from torch.utils.tensorboard import SummaryWriter

class FacialForgeryDataset(Dataset):
    def __init__(self, mask_list, transform, transform_mask):
        self.transform = transform
        self.mask_list = mask_list
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx: int):
        image, mask = self.mask_list[idx]
        label = 0
        image = Image.open(image)
        image = self.transform(image)
        if type(mask)==str:
          mask = Image.open(mask)
          mask = self.transform_mask(mask)
        if 'real' in self.mask_list[idx][0]:
          label = torch.tensor(0)
        else:
          label = torch.tensor(1)
        return {'image': image, 'label': label, 'msk': mask}

class SeparableConv2d(nn.Module):
  def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
    super(SeparableConv2d, self).__init__()
    self.c = nn.Conv2d(c_in, c_in, ks, stride, padding, dilation, groups=c_in, bias=bias)
    self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

  def forward(self, x):
    x = self.c(x)
    x = self.pointwise(x)
    return x

class Block(nn.Module):
  def __init__(self, c_in, c_out, reps, stride=1, start_with_relu=True, grow_first=True):
    super(Block, self).__init__()

    self.skip = None
    self.skip_bn = None
    if c_out != c_in or stride!= 1:
      self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
      self.skip_bn = nn.BatchNorm2d(c_out)

    self.relu = nn.ReLU(inplace=True)

    rep = []
    c = c_in
    if grow_first:
      rep.append(self.relu)
      rep.append(SeparableConv2d(c_in, c_out, 3, stride=1, padding=1, bias=False))
      rep.append(nn.BatchNorm2d(c_out))
      c = c_out

    for i in range(reps - 1):
      rep.append(self.relu)
      rep.append(SeparableConv2d(c, c, 3, stride=1, padding=1, bias=False))
      rep.append(nn.BatchNorm2d(c))

    if not grow_first:
      rep.append(self.relu)
      rep.append(SeparableConv2d(c_in, c_out, 3, stride=1, padding=1, bias=False))
      rep.append(nn.BatchNorm2d(c_out))

    if not start_with_relu:
      rep = rep[1:]
    else:
      rep[0] = nn.ReLU(inplace=False)

    if stride != 1:
      rep.append(nn.MaxPool2d(3, stride, 1))
    self.rep = nn.Sequential(*rep)

  def forward(self, inp):
    x = self.rep(inp)

    if self.skip is not None:
      y = self.skip(inp)
      y = self.skip_bn(y)
    else:
      y = inp

    x += y
    return x

class RegressionMap(nn.Module):
  def __init__(self, c_in):
    super(RegressionMap, self).__init__()
    self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
    self.s = nn.Sigmoid()

  def forward(self, x):
    mask = self.c(x)
    mask = self.s(mask)
    return mask, None

class TemplateMap(nn.Module):
  def __init__(self, c_in, templates):
    super(TemplateMap, self).__init__()
    self.c = Block(c_in, 364, 2, 2, start_with_relu=True, grow_first=False)
    self.l = nn.Linear(364, 10)
    self.relu = nn.ReLU(inplace=True)

    self.templates = templates

  def forward(self, x):
    v = self.c(x)
    v = self.relu(v)
    v = F.adaptive_avg_pool2d(v, (1,1))
    v = v.view(v.size(0), -1)
    v = self.l(v)
    mask = torch.mm(v, self.templates.reshape(10,361))
    mask = mask.reshape(x.shape[0], 1, 19, 19)

    return mask, v

class PCATemplateMap(nn.Module):
  def __init__(self, templates):
    super(PCATemplateMap, self).__init__()
    self.templates = templates

  def forward(self, x):
    fe = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
    fe = torch.transpose(fe, 1, 2)
    mu = torch.mean(fe, 2, keepdim=True)
    fea_diff = fe - mu

    cov_fea = torch.bmm(fea_diff, torch.transpose(fea_diff, 1, 2))
    B = self.templates.reshape(1, 10, 361).repeat(x.shape[0], 1, 1)
    D = torch.bmm(torch.bmm(B, cov_fea), torch.transpose(B, 1, 2))
    eigen_value, eigen_vector = D.symeig(eigenvectors=True)
    index = torch.tensor([9])#.cuda()
    eigen = torch.index_select(eigen_vector, 2, index)

    v = eigen.squeeze(-1)
    mask = torch.mm(v, self.templates.reshape(10, 361))
    mask = mask.reshape(x.shape[0], 1, 19, 19)
    return mask, v

class Xception(nn.Module):
  """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """
  def __init__(self, maptype, templates, num_classes=1000):
    super(Xception, self).__init__()
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(32,64,3,bias=False)
    self.bn2 = nn.BatchNorm2d(64)

    self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
    self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
    self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
    self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

    self.conv3 = SeparableConv2d(1024,1536,3,1,1)
    self.bn3 = nn.BatchNorm2d(1536)

    self.conv4 = SeparableConv2d(1536,2048,3,1,1)
    self.bn4 = nn.BatchNorm2d(2048)

    self.last_linear = nn.Linear(2048, num_classes)

    if maptype == 'none':
      self.map = [1, None]
    elif maptype == 'reg':
      self.map = RegressionMap(728)
    elif maptype == 'tmp':
      self.map = TemplateMap(728, templates)
    elif maptype == 'pca_tmp':
      self.map = PCATemplateMap(728)
    else:
      print('Unknown map type: `{0}`'.format(maptype))
      sys.exit()

  def features(self, input):
    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    mask, vec = self.map(x)
    x = x * mask
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x, mask, vec

  def logits(self, features):
    x = self.relu(features)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    x = self.last_linear(x)
    return x

  def forward(self, input):
    x, mask, vec = self.features(input)
    x = self.logits(x)
    return x, mask, vec

  def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('SeparableConv2d') != -1:
      m.c.weight.data.normal_(0.0, 0.01)
      if m.c.bias is not None:
        m.c.bias.data.fill_(0)
      m.pointwise.weight.data.normal_(0.0, 0.01)
      if m.pointwise.bias is not None:
        m.pointwise.bias.data.fill_(0)
    elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
      m.weight.data.normal_(0.0, 0.01)
      if m.bias is not None:
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
      m.weight.data.normal_(1.0, 0.01)
      m.bias.data.fill_(0)
    elif classname.find('LSTM') != -1:
      for i in m._parameters:
        if i.__class__.__name__.find('weight') != -1:
          i.data.normal_(0.0, 0.01)
        elif i.__class__.__name__.find('bias') != -1:
          i.bias.data.fill_(0)

class Model:
  def __init__(self, maptype='None', templates=None, num_classes=2, load_pretrain=True):
    model = Xception(maptype, templates, num_classes=num_classes)
    if load_pretrain:
      # state_dict = torch.load("/shared/rc/defake/Deepfake-Slayer/pretrained_weights/xception_best.pth")
      state_dict = torch.load("/shared/rc/defake/Deepfake-Slayer/pretrained_weights/xception_best.pth")
      for name, weights in state_dict.items():
        if 'pointwise' in name:
          state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      model.load_state_dict(state_dict, False)
    else:
      model.init_weights()
    self.model = model

  def save(self, epoch, optim, model_dir):
    state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
    torch.save(state, '{0}/{1:06d}.tar'.format(model_dir, epoch))
    print('Saved model `{0}`'.format(epoch))

  def load(self, epoch, model_dir):
    filename = '{0}{1:06d}.tar'.format(model_dir, epoch)
    print('Loading model from {0}'.format(filename))
    if os.path.exists(filename):
      state = torch.load(filename)
      self.model.load_state_dict(state['net'])
    else:
      print('Failed to load model from {0}'.format(filename))

def get_templates():
  templates_list = []
  for i in range(10):
    img = imread('/shared/rc/defake/Deepfake-Slayer/Templates/template{:d}.png'.format(i))
    templates_list.append(transforms.functional.to_tensor(img)[0:1,0:19,0:19])
  templates = torch.stack(templates_list).cuda()
  templates = templates.squeeze(1)
  return templates

MODEL_DIR = '/shared/rc/defake/Deepfake-Slayer/models/train/'
# MODEL_DIR = '/content/gdrive/MyDrive/shared/rc/defake/FaceForensics++_All/FaceForensics++/models/train/'
BACKBONE = 'xcp'
MAPTYPE = 'reg'#'tmp'
BATCH_SIZE = 200
MAX_EPOCHS = 100
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001
MASK_THRESHOLD = 0.5

CONFIGS = {
  'xcp': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.5] * 3, [0.5] * 3]
         }
}

CONFIG = CONFIGS[BACKBONE]
SEED = 1
torch.cuda.manual_seed_all(SEED)
DATA_TEST = None
TEMPLATES = None

if MAPTYPE in ['tmp', 'pca_tmp']:
  TEMPLATES = get_templates()

MODEL_NAME = '{0}_{1}'.format(BACKBONE, MAPTYPE)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'
if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
model = MODEL.model.cuda()
MODEL.load(16,'/shared/rc/defake/Deepfake-Slayer/models-1-16-epochs/FFDmodel')
# state_dict = torch.load("/shared/rc/defake/Deepfake-Slayer/pretrained_weights/new_pretrainedweights.pth")
# state_dict = torch.load("/content/gdrive/MyDrive/FFD Pretrained Weights/new_pretrainedweights.pth")#/content/gdrive/MyDrive/FFD pretrained weights/new_pretrainedweights.pth
# model.load_state_dict(state_dict)
OPTIM = optim.Adam(MODEL.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
LOSS_CSE = nn.CrossEntropyLoss().cuda()
LOSS_L1 = nn.L1Loss().cuda()
MAXPOOL = nn.MaxPool2d(19).cuda()

def calculate_losses(batch):
  img = batch['image'].cuda()
  msk = batch['msk'].cuda()
  lab = batch['label'].cuda()
  x, mask, vec = MODEL.model(img)
  loss_l1 = LOSS_L1(mask, msk)
  loss_cse = LOSS_CSE(x, lab)
  loss = loss_l1 + loss_cse
  pred = torch.max(x, dim=1)[1]
  acc = (pred == lab).float().mean()
  res = { 'lab': lab, 'msk': msk, 'score': x, 'pred': pred, 'mask': mask }
  results = {}
  for r in res:
    results[r] = res[r].squeeze().detach().cpu().numpy()#res[r].squeeze().cpu().detach().numpy()
  return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }, results

def main():
  transform = transforms.Compose([
  transforms.Resize((299, 299)),# Assuming input size is 299x299 based on Xception architecture
  transforms.ToTensor(),
  transforms.Normalize(*[[0.5] * 3, [0.5] * 3])
  ])

  transform_mask = transforms.Compose([
      transforms.Resize((19,19)),
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor()
  ])

  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceSwap.pkl', 'rb') as file:
    FaceSwap_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/Face2Face.pkl', 'rb') as file:
    Face2Face_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceShifter.pkl', 'rb') as file:
    FaceShifter_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/fake_NeuralTextures.pkl', 'rb') as file:
    fake_NeuralTextures_mask = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_yt_train.pkl', 'rb') as file:
    real_yt_train = pickle.load(file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_actors_train.pkl', 'rb') as file:
    real_actors_train = pickle.load(file)

  train_list = FaceSwap_mask['train'] + Face2Face_mask['train'] + FaceShifter_mask['train'] + fake_NeuralTextures_mask['train'] + real_yt_train + real_actors_train
  val_list = FaceSwap_mask['val'] + Face2Face_mask['val'] + FaceShifter_mask['val'] + fake_NeuralTextures_mask['val'] + real_yt_train + real_actors_train
  train_dataset = FacialForgeryDataset(train_list, transform, transform_mask)
  val_dataset = FacialForgeryDataset(val_list, transform, transform_mask)

  batch_size = 128
  shuffle = True
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=4)
  val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=4)
  torch.backends.deterministic = True

  output = '/shared/rc/defake/Deepfake-Slayer/output/train'
  # output = '/content/gdrive/MyDrive/shared/rc/defake/FaceForensics++_All/FaceForensics++/output/train'
  if not os.path.exists(output):
      os.makedirs(output)
  file = open(f'{output}/train.txt','w')
  resultdir = '{0}model'.format(MODEL_DIR)

  num_epochs = 20
  best_val_loss = float('inf')
  patience = 5
  counter = 0
  best_epoch = 0
  model_dir = '/shared/rc/defake/Deepfake-Slayer/models/FFDmodel/'
  best_dir = '/shared/rc/defake/Deepfake-Slayer/models/FFDmodel/best'
  # model_dir = '/content/gdrive/MyDrive/shared/rc/defake/FaceForensics++_All/FaceForensics++/models/FFDmodel'
  if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  if not os.path.exists(best_dir):
      os.makedirs(best_dir)
  writer = SummaryWriter(log_dir='/shared/rc/defake/Deepfake-Slayer/output/logs')
  for e in range(1,num_epochs+1):
    MODEL.model.train()
    for id, batch in enumerate(train_loader):
      losses, results = calculate_losses(batch)
      OPTIM.zero_grad()
      losses['loss'].backward()
      OPTIM.step()
      savemat('{0}_{1}_{2}.mat'.format(resultdir, e, id), results)
      file.write(f"{id} " + " ".join(["{}: {:.3f}".format(key, losses[key].item()) for key in losses]))
      file.write("\n")
      for key, value in losses.items():
        writer.add_scalar(f"Train/{key}", value.item(), e * len(train_loader) + id)

    MODEL.model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for id, batch in enumerate(val_loader):
        losses, results = calculate_losses(batch)
        val_loss += losses['loss']
        for key, value in losses.items():
            writer.add_scalar(f"Validation/{key}", value.item(), e * len(val_loader) + id)
    val_loss /= len(val_loader)
    writer.add_scalar("Validation/total_loss", val_loss, e)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      counter = 0
      best_epoch = e
      MODEL.save(e, OPTIM, best_dir)
      file.write(f'Saving best model at epoch {e}.')
      file.write("\n")
    else:
      counter += 1
      if counter >= patience:
        MODEL.save(e, OPTIM, model_dir)
        file.write(f'Early stopping at epoch {e}. Best epoch: {best_epoch}')
        file.write("\n")
        break
    if(e%2==0):
        file.write(f'Saving model for epoch {e}')
        file.write("\n")
        MODEL.save(e, OPTIM, model_dir)
  file.write('Training complete')
  file.close()
  writer.close()

if __name__ == "__main__":
    main()