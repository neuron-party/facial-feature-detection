import cv2
import torchvision.transforms as T

from core.models import *
from core.networks import *
from core.utils import *


class WebCam:
    def __init__(self, features, path_cascade, path_model, tolerance=10, resolution=(1280, 720)):
        self.features = features
        self.tolerance = tolerance
        self.resolution = resolution
        
        self.device = torch.device('cpu')
        network_resnet = ResNet(34, len(features), in_channels=3).to(self.device)
        self.model_cls = ModelSigmoidClassifier(network_resnet)
        self.model_cls.load(path_model, map_location=self.device)
        self.transforms = T.Compose([
            T.Resize(size=(224, 224))])
        self.cascade = cv2.CascadeClassifier(path_cascade)
        self.cap = cv2.VideoCapture(0)

    def feed(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, self.resolution)
            coords = get_coords(frame, self.cascade, self.tolerance, multi=True)

            if coords is not None:
                # BATCH PREPROCESSING
                x_batch = None
                for coord in coords:
                    x, y, w, h = coord
                    img = cv2.cvtColor(frame[y:y + h, x:x + h], cv2.COLOR_BGR2RGB)
                    img = img.transpose(2, 0, 1)
                    x_ = torch.tensor(img, device=self.device, dtype=torch.float32).unsqueeze(0)
                    x_ = self.transforms(x_)
                    if x_batch is None:
                        x_batch = x_
                    else:
                        x_batch = torch.cat([x_batch, x_])

                # BATCH PREDICTIONS
                y_pred = self.model_cls.network(x_batch)
                probs = torch.sigmoid(y_pred).cpu().detach().numpy()

                # OUTPUT
                for i in range(probs.shape[0]):
                    x, y, w, h = coords[i]
                    labels = {self.features[j]:probs[i][j] for j in range(len(probs[i]))}

                    cv2.rectangle(
                        img=frame,
                        pt1=(x, y),
                        pt2=(x + w, y + h),
                        color=(255, 255, 0), # BGR
                        thickness=1)

                    d = 13
                    for k, v in labels.items():
                        cv2.putText(
                            img=frame,
                            text=f'{k}:{v:3.2f}',
                            org=(x, y + h + d),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=.5,
                            color=(0, 255, 0), # BGR
                            thickness=1)
                        d += 13

            ret, jpeg = cv2.imencode('.jpg', frame)
            output = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n\r\n')
