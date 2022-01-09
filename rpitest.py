import cv2 as cv
import numpy as np
import time
from gpiozero import Buzzer, Servo

bz = Buzzer(3)
servo = Servo(4)

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None
count = 1

# Read Class Names for detection
classes = open('coco.names').read().strip().split('\n')
# Read Class Names to look out for
danger = open('danger.txt').read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv.dnn.readNet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def load_image():
    global img, frame, outputs, ln, count

    print(count)

    img = frame.copy()

    blob = cv.dnn.blobFromImage(
        img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time() - t0
    print('time=', t)

    outputs = np.vstack(outputs)

    post_process(img, outputs, 0.2)
    cv.imwrite('./result/r{0}.jpg'.format(str(count)), frame)
    count += 1


def post_process(img, outputs, conf):
    global bz

    H, W = img.shape[:2]

    classIDs = []
    objects = []
    common = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            classIDs.append(classID)
    for ID in classIDs:
        objects.append(classes[ID])
    objects = np.unique(objects)
    print(objects)

    common = np.intersect1d(objects, danger)

    if(len(common)):
        # Raise Alarm
        bz.on()
        print("ALARM")


cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    t0 = time.time()
    val = -10
    add = 2

    frame = None
    success, frame = cap.read()
    if success:
        load_image()
    else:
        print('Error in Camera')  # For debugging purposes
        continue

    servo.value = val/10.0

    while(time.time()-t0 < 20):
        continue

    bz.off()
    val += add
    print(val)

    if(val == 10 or val == -10):
        add *= -1

    print('time taken:   ')
    print(time.time()-t0)
