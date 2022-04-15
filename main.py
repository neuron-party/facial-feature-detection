import torchvision.transforms as T

from core.networks import *
from core.models import *
from core.utils import *


dir_load = './resources/models/'
path_cascade = './resources/haarcascade_frontalface_default.xml'
features = ['Attractive', 'Bags_Under_Eyes', 'Bangs', 'Chubby', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Smiling', 'Wearing_Lipstick', 'Young']

device = torch.device('cpu')
# cae_latent_dim = 32
# cae_stride = 2
# x_shape = (3, 224, 224)
resnet_model_no = 34


# network_cae = ConvAutoencoder(cae_latent_dim, *x_shape, stride=cae_stride).to(device)
network_resnet = ResNet(resnet_model_no, len(features), in_channels=3).to(device)

# model_cae = ModelCAE(network_cae)
model_cls = ModelSigmoidClassifier(network_resnet)

# model_cae.load(dir_load + 'cae.pth', map_location=device)
model_cls.load(dir_load + 'cls.pth', map_location=device)

transforms = T.Compose([
    T.Resize(size=(224, 224))])

cascade = cv2.CascadeClassifier(path_cascade)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
resolution = (1280, 720)
fps = 60.0
out = cv2.VideoWriter('./demo/output.avi', fourcc, fps, resolution)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, resolution)
    coords = get_coords(frame, cascade, 10, multi=True)

    if coords is not None:
        # BATCH PREPROCESSING
        x_batch = None
        for coord in coords:
            x, y, w, h = coord
            img = cv2.cvtColor(frame[y:y + h, x:x + h], cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            x_ = torch.tensor(img, device=device, dtype=torch.float32).unsqueeze(0)
            x_ = transforms(x_)
            if x_batch is None:
                x_batch = x_
            else:
                x_batch = torch.cat([x_batch, x_])
        
        # BATCH PREDICTIONS
#         x_pred = model_cae.network(x_batch)
        y_pred = model_cls.network(x_batch)
        probs = torch.sigmoid(y_pred).cpu().detach().numpy()
        
        # OUTPUT
        for i in range(probs.shape[0]):
            x, y, w, h = coords[i]
            labels = {features[j]:probs[i][j] for j in range(len(probs[i]))}

            cv2.rectangle(
                img=frame,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(255, 255, 0), # BGR
                thickness=1)
            
            # SELECT SPECIFIC FEATURES
            f = ['Smiling', 'Attractive', 'Bangs']
            labels = {x:labels[x] for x in f}
            
            d = 30
            for k, v in labels.items():
                cv2.putText(
                    img=frame,
                    text=f'{k}:{v:3.2f}',
                    org=(x, y + h + d),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 255, 0), # BGR
                    thickness=2)
                d += 30

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
