import cv2
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter

from model import FaceNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FaceNet(device)
# cap = cv2.VideoCapture('sample_no_audio_10fps.mp4')
cap = cv2.VideoCapture('sample_3_fps10.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print('cap :',cap.isOpened())
print('total frames :', count)
print('fps :', fps)
print('size :', f"({height},{width})")

entire_imgs = []
entire_embs = []
for i in tqdm(range(count)):
    if not cap.isOpened():
        break
    ret,frame = cap.read()
    if i > 400:
        break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    start_time = time()
    boxes, cropped_img_list = model.detect(img)
    embs = model.embedding(cropped_img_list)
    print('process time :', time() - start_time)

    entire_imgs.extend([np.array(img.resize((160,160))).transpose((2,0,1)) / 255 for img in cropped_img_list])
    entire_embs.extend(embs)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(f'imgs : {len(entire_imgs)}, embs : {len(entire_embs)}')
print(np.array(entire_imgs).shape)

writer = SummaryWriter('runs')
mat = torch.stack(entire_embs,dim=0).squeeze().cpu()
writer.add_embedding(mat, label_img=np.array(entire_imgs))
writer.close()

