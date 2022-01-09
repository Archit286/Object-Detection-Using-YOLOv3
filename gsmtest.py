import cv2 as cv
import numpy as np
import time
from serial import Serial

gsm = Serial("/dev/ttyUSB0", 9600, timeout=0.5)
gsm.flush()

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None
mobile = '##########'  # Mobile Number to send the SMS

# Read Class Names for detection
classes = open('coco.names').read().strip().split('\n')
# Read Class Names to look out for
danger = open('danger.txt').read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv.dnn.readNet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


def load_image(path):
    global img, img0, outputs, ln

    img0 = cv.imread(path)
    img = img0.copy()

    blob = cv.dnn.blobFromImage(
        img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time() - t0
    print('time=', t)

    outputs = np.vstack(outputs)

    post_process(img, outputs, 0.2)


def post_process(img, outputs, conf):
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
        sendSms('There is an intrusion in the farm')
        print("ALARM")


def sendSms(msg):
    print("Sending SMS\n")
    gsm.write(b'AT+CMGF=1\r\n')
    time.sleep(0.5)
    gsm.write(b'AT+CMGS=\"')
    gsm.write(mobile.encode())
    gsm.write(b'\"\r\n')
    time.sleep(0.5)
    data = msg
    gsm.write(data.encode())
    gsm.write(b'\x1A')
    time.sleep(3)


for j in range(1, 26):
    t0 = time.time()
    print(j)
    load_image('test/s'+str(j)+'.jpg')

    while(time.time()-t0 < 20):
        continue

    print('time taken:   ')
    print(time.time()-t0)
