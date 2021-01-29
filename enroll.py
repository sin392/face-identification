import cv2
from PIL import Image
import torch
from tqdm import tqdm
from glob import glob
import os
import pandas as pd

from model import FaceNet

def extract_lab(path, pos=2):
    for i in range(pos-1):
        path = os.path.dirname(path)
    label = os.path.basename(path)
    return label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FaceNet(device)

img_paths = glob('enroll/*/*')

data = []
for i,path in enumerate(tqdm(img_paths)):
    img = Image.open(path).convert('RGB').resize((160,160))
    boxes, cropped_img_list = model.detect(img)
    embs = model.embedding(cropped_img_list)
    for emb in embs:
        data.append((path, extract_lab(path), emb.squeeze().cpu()))

df = pd.DataFrame(data, columns=['path', 'label','emb'])
print(df)
df.to_pickle('enroll_df.pkl')