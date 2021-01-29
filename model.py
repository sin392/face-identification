from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from torchvision.transforms import ToTensor
import torch

def compute_area(box):
    x1,y1,x2,y2 = box
    area = (x2 - x1) * (y2 - y1)
    return area

class FaceNet:
    def __init__(self, device):
        self.device = device
        self.input_size = (160,160)
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(
            pretrained='vggface2', device=device).eval()

    def detect(self, img, area_th=None):
        # 最小面積はMTCNNのmin_face_size渡せば済む話では？
        if area_th == None:
            width, height = img.size
            area_th = (width // 10) * (height // 10)
        with torch.no_grad():
            boxes, _ = self.mtcnn.detect(img)
            # drop small box
        if isinstance(boxes, np.ndarray):
            boxes = [box for box in boxes if compute_area(box) > area_th]
            cropped_img_list = [img.crop(box) for box in boxes]
        else:
            boxes, cropped_img_list = [], []
        return boxes, cropped_img_list

    def embedding(self, cropped_img_list):
        embs = []
        for img in cropped_img_list:
            resized_img = img.resize(self.input_size)
            img_tensor = ToTensor()(resized_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embs.append(self.resnet(img_tensor))
        return embs

if __name__ == '__main__':
    from PIL import Image
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device',device)
    img = Image.open('sample.png').convert('RGB')
    # print(img)
    model = FaceNet(device)
    boxes, cropped_img_list = model.detect(img)
    for i, cropped_img in enumerate(cropped_img_list):
        print(model.embedding(cropped_img))