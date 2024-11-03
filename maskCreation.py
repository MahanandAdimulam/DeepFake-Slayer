from pathlib import Path
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from imageio import imread
import glob
import pickle

def find_filenames(file_dir_path, file_pattern):
    return list(file_dir_path.glob(file_pattern))

def create_maskpath(filepath):#defake/faceSwap
    folders = os.listdir(filepath)#train, test, val
    folders = [file for file in folders if not file.endswith('.txt')]
    mask_dict = {}
    mask = []
    for folder in folders:
      for sub_folder in os.listdir(filepath+'/'+folder):
        mask.append(filepath+'/'+folder+'/'+sub_folder+'/')#000_003, 000_004, ...
    for file_path in mask:
      data_path = Path(file_path)
      frame_filenames = find_filenames(data_path,'*.jpeg')
      frame_filenames = [str(f) for f in frame_filenames]
      frame_filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
      mask_dict[file_path.split('/')[-2].split('.')[0]] = frame_filenames
    return mask_dict

def create_realpath(filepath):#defake/faceSwap
    folders = os.listdir(filepath)
    videos = []
    for folder in folders:
      videos.append(filepath+'/'+folder+'/')
    filenames = []
    for file_path in videos:
        data_path = Path(file_path)
        real_frame_filenames = find_filenames(data_path,'*.jpeg')
        real_frame_filenames = [(str(f),torch.zeros(1,19,19)) for f in real_frame_filenames]
        filenames += real_frame_filenames
    return filenames

def create_fakeImage_mask_pair(fake_image_dict, real_image_dict, name):
  not_found = {}
  mask_dict = {'test':[], 'train':[], 'val':[]}
  output = f"/shared/rc/defake/Deepfake-Slayer/output/masksPair/{name}"
#   output = f"/content/gdrive/MyDrive/shared/rc/defake/FaceForensics++_All/FaceForensics++/output/maskPair/{name}"
  if not os.path.exists(output):
        os.makedirs(output)
  file = open(f'{output}/{name}.txt','w')
  transform_image = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()])
  transform_mask = transforms.Grayscale()
  directory_not_found = []
  for key in fake_image_dict:
    if key.split('_')[0] in real_image_dict.keys():
      fake_filenames = fake_image_dict[key]
      real_filenames = real_image_dict[key.split('_')[0]]
      directory = f"/shared/rc/defake/Deepfake-Slayer/masks/{key}"
    #   directory = f"/content/gdrive/MyDrive/shared/rc/defake/FaceForensics++_All/FaceForensics++/mask/{key}"
      if not os.path.exists(directory):
        os.makedirs(directory)
      file.write(f"For {key} directory in fake and {key.split('_')[0]} directory in real\n")
      frame_not_found = 0
      frame_not = []
      for i in range(len(fake_filenames)):
        fake_filename = fake_filenames[i].split('/')[-1]
        real_filename = [i.split('/')[-1] for i in real_filenames]
        if fake_filename in real_filename:
          idx = real_filename.index(fake_filename)
          real_image = Image.open(real_filenames[idx])
          real_image = transform_image(real_image)
          fake_image = Image.open(fake_filenames[i])
          fake_image = transform_image(fake_image)
          mask = fake_image - real_image
          mask = transform_mask(mask)
          threshold= 0.1#10e-10
          binary = torch.where(mask >= threshold, torch.tensor(1), torch.tensor(0))
          binarr = binary.squeeze().numpy()*255
          binary_image = Image.fromarray(binarr.astype('uint8'))
          plt.imsave(f"{directory}/{fake_filename}", binary_image, cmap='gray', format='jpeg')
          if 'test' in fake_filenames[i]:
            mask_dict['test'].append((fake_filenames[i], f"{directory}/{fake_filename}"))
            # break
          elif 'train' in fake_filenames[i]:
            mask_dict['train'].append((fake_filenames[i], f"{directory}/{fake_filename}"))
            # break
          if 'val' in fake_filenames[i]:
            mask_dict['val'].append((fake_filenames[i], f"{directory}/{fake_filename}"))
            # break
        else:
          frame_not.append(f"{fake_filenames[i]}")
          file.write(f"{fake_filename} frame in {key} not found in {key.split('_')[0]}\n")
      not_found[key] = frame_not
    else:
      file.write(f'{key} not found in real\n')
      directory_not_found.append(key)
  not_found['directory_not_found'] = directory_not_found
  file.close()
  return mask_dict, not_found

def main():
  # fake_FaceSwap = create_maskpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/FaceSwap')
  # fake_Face2Face = create_maskpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/Face2Face')
  # fake_FaceShifter = create_maskpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/FaceShifter')
  # fake_NeuralTextures = create_maskpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/fake/NeuralTextures')
  # real_dict = create_maskpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube')
  real_yt_test = create_realpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/test')
  real_actors_test = create_realpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/misc/actors/test')

  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_dict.pkl', 'wb') as pickle_file:
  #   pickle.dump(real_dict, pickle_file)
  # real_yt_train = create_realpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/real/youtube/train')
  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_yt_train.pkl', 'wb') as pickle_file:
  #   pickle.dump(real_yt_train, pickle_file)
  # real_actors_train = create_realpath('/shared/rc/defake/FaceForensics++_All/FaceForensics++/misc/actors/train')
  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_actors_train.pkl', 'wb') as pickle_file:
  #   pickle.dump(real_actors_train, pickle_file)
  # FaceSwap_mask, FaceSwap_not_found = create_fakeImage_mask_pair(fake_FaceSwap, real_dict, 'FaceSwap_Not_Found')
  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceSwap.pkl', 'wb') as pickle_file:
  #   pickle.dump(FaceSwap_mask, pickle_file)
  # Face2Face_mask, Face2face_not_found = create_fakeImage_mask_pair(fake_Face2Face, real_dict, "Face2Face_Not_Found")
  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/Face2Face.pkl', 'wb') as pickle_file:
  #   pickle.dump(Face2Face_mask, pickle_file)
  # FaceShifter_mask, FaceShifter_not_found = create_fakeImage_mask_pair(fake_FaceShifter, real_dict, "FaceShifter_Not_Found")
  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/FaceShifter.pkl', 'wb') as pickle_file:
  #   pickle.dump(FaceShifter_mask, pickle_file)
  # fake_NeuralTextures_mask, NeuralTextures_not_found = create_fakeImage_mask_pair(fake_NeuralTextures, real_dict, "NeuralTextures_Not_Found")
  # with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/fake_NeuralTextures.pkl', 'wb') as pickle_file:
  #   pickle.dump(fake_NeuralTextures_mask, pickle_file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_yt_test.pkl', 'wb') as pickle_file:
    pickle.dump(real_yt_test, pickle_file)
  with open('/shared/rc/defake/Deepfake-Slayer/pickel_file/real_actors_test.pkl', 'wb') as pickle_file:
    pickle.dump(real_actors_test, pickle_file)

if __name__ == "__main__":
    main()