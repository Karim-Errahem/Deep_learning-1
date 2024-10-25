import cv2
import tensorflow as tf

# Load class names from the specified file
classnames = []
classfile = 'files/thing.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model
p = 'C:/Users/HP/Desktop/deep_learning1/files/frozen_inference_graph.pb'
v = 'C:/Users/HP/Desktop/deep_learning1/files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
net = cv2.dnn.DetectionModel(v, p)
net.setInputSize(320, 230)         
net.setInputScale(1.0 / 127.5)    
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    im = cv2.resize(frame, (900, 576))  # Resize the frame
    classIds, confs, bbox = net.detect(im, confThreshold=0.5)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(im, box, color=(0, 255, 0), thickness=3)
        cv2.putText(im, classnames[classId - 1],
                    (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

    cv2.imshow('Object Detection', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break


cap.release()

