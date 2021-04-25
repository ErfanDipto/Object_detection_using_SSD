import torch
import imageio
import cv2
from torch.autograd import Variable
from ssd import build_ssd
from data import BaseTransform, VOC_CLASSES as labelmap


# defining a function that will do the detection
def detects(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurrences, (score, x0, y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= .06:
            pt = (detections[0, i, j, 0] * scale).numpy()
            cv2.rectangle(frame, pt1=(int(pt[0]), int(pt[1])), pt2=(int(pt[2]), int(pt[3])),
                          color=(255, 0, 0), thickness=2)
            cv2.putText(frame, text=labelmap[i-1], org=(int(pt[0]), int(pt[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            j += 1
    return frame


# creating the ssd neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc:storage))

# creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# doing some object detection on a video
reader = imageio.get_reader('epic_horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('Detected_horses.mp4', fps=fps)
for iters, frame in enumerate(reader):
    frame = detects(frame, net.eval(), transform)
    writer.append_data(frame)
    print(iters)
writer.close()
