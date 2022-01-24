import numpy as np
import pandas as pd
import glob
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

tqdm.pandas()


class CelebA(torch.utils.data.Dataset):
    def __init__(self, dir_data, path_metadata, features, transforms=None):
        super().__init__()
        self.data_paths = glob.glob(dir_data + '*.jpg')
        self.data_paths.sort()
        self.metadata = pd.read_csv(path_metadata)
        self.features = features
        
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(224, 224)),
                T.ToTensor()])
        else:
            self.transforms = transforms
        
        print('Processing Labels...')
        self.metadata['multihotencoding'] = self.metadata.progress_apply(self._get_multihotencoding, axis=1)
        
        self.name_to_idx = {self.metadata['image_id'].iloc[i]:i for i in self.metadata.index}
        
    def __len__(self):
        return len(self.data_paths)

    def _get_multihotencoding(self, row):
        v = row[self.features].to_numpy() == 1
        return np.array(v, dtype=np.float32)

    def __getitem__(self, i):
        x = self.transforms(Image.open(self.data_paths[i]))
        
        file_name = self.data_paths[i].split('\\')[-1]
        idx = self.name_to_idx[file_name]
        y = self.metadata['multihotencoding'].iloc[idx]

        return idx, x, y