import cv2
from PIL import Image, ImageDraw
import torch
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from time import time
from model import FaceNet, compute_area

def draw_bbox(img, boxes, labels, fps=None):
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    for box,label in zip(boxes,labels):
        if label == 'unknown':
            color = (0,0,255)
        else:
            if label == 'Identity-A':
                color = (255,0,0)
            elif label == 'Identity-B':
                color = (0,255,0)
            
        draw.rectangle(box, outline=color)
        draw.text((box[0]+5, box[1]+5), label, fill=color)
    if fps != None:
        draw.text((10,0),f"FPS : {fps}")
    return new_img

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    vec1_norm = np.linalg.norm(vec1, ord=2)
    vec2_norm = np.linalg.norm(vec2, ord=2)
    return dot / (vec1_norm * vec2_norm)

class User:
    def __init__(self, label, fps):
        self.label = label
        self.start = datetime.datetime.now()
        self.end = None
        # 約５秒消えたら終了
        self.limit = int(5 * fps)
        self.count = 0

        self.end_flag = False

    def update(self, labels):
        if self.label not in labels:
            if self.count == 0:
                self.end = datetime.datetime.now()

            self.count += 1
            print(self.label, self.limit, self.count)        
            if self.count > self.limit:
                self.end_flag = True
        else:
            self.count = 0

class StayTimeObserver:
    def __init__(self, fps):
        self.stack = []
        self.fps = fps
        self.columns = ['label', 'start', 'end']
        self.result_df = pd.DataFrame(index=[], columns=self.columns)
        self.save_path = 'time_stamp.csv'
        self.result_df.to_csv(self.save_path, index=False)
    
    def add(self, label):
        self.stack.append(User(label, self.fps))

    def show_labels(self):
        return [x.label for x in self.stack]

    def get(self, label):
        idx = self.show_labels().index(label)
        return self.stack[idx]

    def remove(self, label):
        idx = self.show_labels().index(label)
        obj = self.stack[idx]
        record = pd.DataFrame({ key:val for key,val in zip(self.columns,[label, obj.start, obj.end])}, index=[0],columns=self.columns)
        record.to_csv(self.save_path, mode='a', header=False, index=False)
        self.result_df.append(record, ignore_index=True)
        del self.stack[idx]

    def update(self, labels):
        for label in labels:
            if label != 'unknown' and label not in self.show_labels():
                self.add(label)
        for obj in self.stack:
            obj.update(labels)
            if obj.end_flag == True:
                self.remove(obj.label)

    def show_log(self):
        print(self.result_df)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FaceNet(device)
enroll_df = pd.read_pickle('enroll_df.pkl')
print(enroll_df)

# img = Image.open('sample.png').convert('RGB')
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('sample_3_fps10.mp4')

cap.set(cv2.CAP_PROP_FPS, 10)
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
area_th = (width // 10) * (height // 10)

print('cap :',cap.isOpened())
print('total frames :', count)
print('fps :', fps)
print('size :', f"({height},{width})")
print('area_th :', area_th)

observer = StayTimeObserver(fps)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("output.mp4", fmt, fps, (height, width))

for i in tqdm(range(count)):
    if not cap.isOpened():
        break
    ret,frame = cap.read()
    # if i <= 280:
    #     continue
    start_time = time()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, cropped_img_list = model.detect(img, area_th)
    embs = model.embedding(cropped_img_list)
    labels = []
    for j,emb in enumerate(embs):
        similarities = [cosine_similarity(emb.cpu(), x) for x in enroll_df['emb']]
        smallest_idx = np.argmax(similarities)
        smallest_cs = similarities[smallest_idx]
        thresh = 0.7
        predicted_label = enroll_df['label'][smallest_idx] if smallest_cs >= thresh else 'unknown'
        labels.append(predicted_label)
        print(i, predicted_label, smallest_cs, compute_area(boxes[j]))

        # x1,y1,x2,y2 = [int(el) if el > 0 else 0 for el in boxes[j]]
        # print(x1,y1,x2,y2)
        # cv2.imwrite(f'frames/frame{i}-face{j}.jpg', frame[y1:y2, x1:x2])

    observer.update(labels)
    print('process time :', time() - start_time)

    drawen_img = draw_bbox(img, boxes, labels, fps)
    result = cv2.cvtColor(np.array(drawen_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('frame',result)
    writer.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()

observer.show_log()