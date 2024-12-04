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
import torch.optim as optim
from imageio import imread
import glob
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import auc

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
      state_dict = torch.load('/shared/rc/defake/Deepfake-Slayer/pretrained_weights/xception_best.pth')
      # state_dict = torch.load("/content/gdrive/MyDrive/FFD Pretrained Weights/new_pretrainedweights.pth")#, map_location=torch.device('cpu')
      for name, weights in state_dict:
        if 'pointwise' in name:
          state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      del state_dict['fc.weight']
      del state_dict['fc.bias']
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
      print('Model loaded from {0}'.format(filename))
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

MAPTYPE = 'reg'#'tmp'

SEED = 1
torch.cuda.manual_seed_all(SEED)

DATA_TEST = None

TEMPLATES = None
if MAPTYPE in ['tmp', 'pca_tmp']:
  TEMPLATES = get_templates()

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)
MODEL.load(18,'/shared/rc/defake/Deepfake-Slayer/models_binary/FFDmodel/best/')

model = MODEL.model.cuda()
# state_dict = torch.load("/content/gdrive/MyDrive/FFD pretrained weights/new_pretrainedweights.pth")#/content/gdrive/MyDrive/FFD pretrained weights/new_pretrainedweights.pth
# state_dict = torch.load("/shared/rc/defake/Deepfake-Slayer/pretrained_weights/new_pretrainedweights.pth")
# model.load_state_dict(state_dict)

LOSS_CSE = nn.CrossEntropyLoss().cuda()
LOSS_L1 = nn.L1Loss().cuda()
MAXPOOL = nn.MaxPool2d(19).cuda()

def iou_loss(predicted_mask, ground_truth_mask, smooth=1e-6):
  predicted_mask = (predicted_mask >= 0.3).astype(float)
  ground_truth_mask = (ground_truth_mask >= 0.3).astype(float)

  predicted_mask = predicted_mask.view(predicted_mask.size(0), -1)
  ground_truth_mask = ground_truth_mask.view(ground_truth_mask.size(0), -1)

  intersection = (predicted_mask * ground_truth_mask).sum(dim=1)
  union = predicted_mask.sum(dim=1) + ground_truth_mask.sum(dim=1) - intersection

  iou = (intersection + smooth) / (union + smooth)

  return 1 - iou.mean()

def calculate_losses(batch):
  img = batch['image'].cuda()
  msk = batch['msk'].cuda()
  lab = batch['label'].cuda()
  x, mask, vec = MODEL.model(img)
  loss_iou = iou_loss(mask, msk)
  loss_l1 = LOSS_L1(mask, msk)
  loss_cse = LOSS_CSE(x, lab)
  loss = loss_cse + 0.3*loss_l1 + 0.7*loss_iou
  pred = torch.max(x, dim=1)[1]
  acc = (pred == lab).float().mean()
  res = { 'lab': lab, 'img': img, 'msk': msk, 'score': x, 'pred': pred, 'mask': mask }
  results = {}
  for r in res:
    results[r] = res[r].squeeze().cpu().numpy()
  return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'loss_iou': loss_iou, 'acc': acc }, results

MASK_THRESHOLD = 0.5

def compute_result_file(rfn):
    rf = loadmat(rfn)
    # print(rf)
    res = {}
    for r in ['lab', 'msk', 'score', 'pred', 'mask']:
      res[r] = rf[r].squeeze()
    return res

def iou_accuracy(predicted_mask, ground_truth_mask,smooth=1e-6):
    predicted_mask = (predicted_mask >= 0.3).astype(float)
    ground_truth_mask = (ground_truth_mask >= 0.3).astype(float)

    predicted_mask = predicted_mask.reshape(predicted_mask.shape[0], -1)
    ground_truth_mask = ground_truth_mask.reshape(ground_truth_mask.shape[0], -1)

    # Calculate intersection and union
    intersection = (predicted_mask * ground_truth_mask).sum(axis=1)
    union = predicted_mask.sum(axis=1) + ground_truth_mask.sum(axis=1) - intersection

    # IoU for each image in the batch
    iou = (intersection + smooth) / (union + smooth)

    # Return mean IoU (accuracy) across the batch
    return iou.mean()
    # return iou[predicted_labels == 1].mean() if np.any(predicted_labels == 1) else 0.0

def iou_accuracy_fake(predicted_mask, ground_truth_mask, predicted_labels,smooth=1e-6):
    predicted_mask = (predicted_mask >= 0.3).astype(float)
    ground_truth_mask = (ground_truth_mask >= 0.3).astype(float)

    predicted_mask = predicted_mask.reshape(predicted_mask.shape[0], -1)
    ground_truth_mask = ground_truth_mask.reshape(ground_truth_mask.shape[0], -1)

    # Calculate intersection and union
    intersection = (predicted_mask * ground_truth_mask).sum(axis=1)
    union = predicted_mask.sum(axis=1) + ground_truth_mask.sum(axis=1) - intersection

    # IoU for each image in the batch
    iou = (intersection + smooth) / (union + smooth)
    if np.isscalar(predicted_labels):
        predicted_labels = np.array([predicted_labels])

    iou = np.zeros(predicted_labels.shape[0])

    # Calculate IoU only for images with predicted label 1
    for i in range(predicted_labels.shape[0]):
        if predicted_labels[i] == 1:
            iou[i] = (intersection[i] + smooth) / (union[i] + smooth)
        else:
            iou[i] = 0.0  # IoU is set to 0 if predicted label is 0

    # Return mean IoU (accuracy) across the batch
    return iou[predicted_labels == 1].mean() if np.any(predicted_labels == 1) else 0.0

def eval():
    for color in ['red', 'green', 'blue']:
        # output = "output/eval"
        output = "/shared/rc/defake/Deepfake-Slayer/output/eval"
        if not os.path.exists(output):
            os.makedirs(output)
        file = open(f'{output}/eval_rgb_{color}.txt','w')

        # Compile the results into a single variable for processing
        TOTAL_RESULTS = {}
        RESDIR = f'/shared/rc/defake/Deepfake-Slayer/models_binary/test_rgb_{color}/xcp_reg/'
        print(RESDIR)
        RESFILENAMES = glob.glob(RESDIR + '*.mat')
        for rfn in RESFILENAMES:
            rf = compute_result_file(rfn)
            for r in rf:
                if r not in TOTAL_RESULTS:
                    TOTAL_RESULTS[r] = rf[r]
                else:
                    TOTAL_RESULTS[r] = np.concatenate([TOTAL_RESULTS[r], rf[r]], axis=0)
        print(TOTAL_RESULTS.keys())
        file.write('Found {0} total images with scores.\n'.format(TOTAL_RESULTS['lab'].shape[0]))
        file.write('  {0} results are real images.\n'.format((TOTAL_RESULTS['lab'] == 0).sum()))
        file.write('  {0} results are fake images.\n'.format((TOTAL_RESULTS['lab'] == 1).sum()))

        # Compute the performance numbers
        PRED_ACC = (TOTAL_RESULTS['lab'] == TOTAL_RESULTS['pred']).astype(np.float32).mean()
        MASK_ACC = ((TOTAL_RESULTS['mask'] >= MASK_THRESHOLD) == (TOTAL_RESULTS['msk'] >= MASK_THRESHOLD)).astype(np.float32).mean()
        MASK_ACC_IOU = iou_accuracy(TOTAL_RESULTS['mask'], TOTAL_RESULTS['msk'])
        MASK_ACC_IOU_FAKE = iou_accuracy(TOTAL_RESULTS['mask'], TOTAL_RESULTS['msk'],TOTAL_RESULTS['pred'])
        # Calculate confusion matrix components
        TP = np.sum((TOTAL_RESULTS['lab'] == 1) & (TOTAL_RESULTS['pred'] == 1))  # True Positives
        FP = np.sum((TOTAL_RESULTS['lab'] == 0) & (TOTAL_RESULTS['pred'] == 1))  # False Positives
        FN = np.sum((TOTAL_RESULTS['lab'] == 1) & (TOTAL_RESULTS['pred'] == 0))  # False Negatives

        # Calculate Precision and Recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        FPR, TPR, THRESH = metrics.roc_curve(TOTAL_RESULTS['lab'], TOTAL_RESULTS['score'][:,1], drop_intermediate=False)
        AUC = auc(FPR, TPR)
        FNR = 1 - TPR
        EER = FNR[np.argmin(np.absolute(FNR - FPR))]
        TPR_AT_FPR_NOT_0 = TPR[FPR != 0].min()
        TPR_AT_FPR_THRESHOLDS = {}
        for t in range(-1, -7, -1):
            thresh = 10**t
            TPR_AT_FPR_THRESHOLDS[thresh] = TPR[FPR <= thresh].max()

        # Print out the performance numbers
        file.write('Prediction Accuracy: {0:.4f}\n'.format(PRED_ACC))
        file.write('Mask Accuracy: {0:.4f}\n'.format(MASK_ACC))
        file.write('Mask Accuracy IOU: {0:.4f}\n'.format(MASK_ACC_IOU))
        file.write('Mask Accuracy IOU FAKE: {0:.4f}\n'.format(MASK_ACC_IOU_FAKE))
        file.write('Precision: {0:.4f}\n'.format(precision))
        file.write('Recall: {0:.4f}\n'.format(recall))
        file.write('AUC: {0:.4f}\n'.format(AUC))
        file.write('EER: {0:.4f}\n'.format(EER))
        file.write('Minimum TPR at FPR != 0: {0:.4f}\n'.format(TPR_AT_FPR_NOT_0))

        file.write('TPR at FPR Thresholds:\n')
        for t in TPR_AT_FPR_THRESHOLDS:
            file.write('  {0:.10f} TPR at {1:.10f} FPR\n'.format(TPR_AT_FPR_THRESHOLDS[t], t))

        file.close()
        fig = plt.figure()
        plt.plot(FPR, TPR)
        plt.xlabel('FPR (%)')
        plt.ylabel('TPR (%)')
        plt.xlim([0.5,1])
        plt.ylim([0, 1])
        plt.grid()
        plt.savefig(output+f'/AUC_rgb_{color}.png')

def main():
  eval()
  print(' Forced RGB Values Evaluation complete')

if __name__ == "__main__":
    main()