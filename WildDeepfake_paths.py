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

def create_realpath(filepath):#defake/faceSwap
    folders = os.listdir(filepath)
    videos = []

    for folder in folders:
      for sub_folder in os.listdir(filepath+'/'+folder):
          videos.append(filepath+'/'+folder+'/'+sub_folder+'/')

    filename_dict = {}
    for file_path in videos:
        data_path = Path(file_path)
        real_frame_filenames = find_filenames(data_path,'*.png')
        real_frame_filenames = [str(f) for f in real_frame_filenames]
        if file_path.split('/')[-3] not in filename_dict.keys():
          filename_dict[file_path.split('/')[-3]] = real_frame_filenames
        else:
          filename_dict[file_path.split('/')[-3]] += real_frame_filenames

    return filename_dict

if not os.path.exists('/shared/rc/defake/Deepfake-Slayer/WildDeepfake_pickel_file/'):
        os.makedirs('/shared/rc/defake/Deepfake-Slayer/WildDeepfake_pickel_file')

fake_Wilddeepfake = create_realpath('/shared/rc/defake/WildDeepfakes/fake')
with open('/shared/rc/defake/Deepfake-Slayer/WildDeepfake_pickel_file/fake_Wilddeepfake.pkl', 'wb') as pickle_file:
  pickle.dump(fake_Wilddeepfake, pickle_file)

real_Wilddeepfake = create_realpath('/shared/rc/defake/WildDeepfakes/real')
with open('/shared/rc/defake/Deepfake-Slayer/WildDeepfake_pickel_file/real_Wilddeepfake.pkl', 'wb') as pickle_file:
  pickle.dump(real_Wilddeepfake, pickle_file)

# real_Wilddeepfake_test = create_realpath('/content/gdrive/MyDrive/shared/rc/defake/WildDeepfakes/real/test')
# with open('/content/gdrive/MyDrive/shared/rc/defake/Deepfake-Slayer/WildDeepfake_pickel_file/real_Wilddeepfake_test.pkl', 'wb') as pickle_file:
#   pickle.dump(real_Wilddeepfake_test, pickle_file)