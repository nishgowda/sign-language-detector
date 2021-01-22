#!/usr/bin/env python3
import numpy as np
import cv2
import torch

from model import CNN
import sys

cap = cv2.VideoCapture(0)

cap.set(3, 700)
cap.set(4, 480)

model_name = sys.argv[1]

model = CNN()
model.load_state_dict(torch.load(f"models/{model_name}"))
model.eval()

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

print("Press q or Q to quit")
while True:
    ret, frame = cap.read()

    img = frame[20:250, 20:250]

    res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res1 = np.reshape(res, (1, 1, 28, 28)) / 255
    res1 = torch.from_numpy(res1)
    res1 = res1.type(torch.FloatTensor)

    out = model(res1)
    
    probs, label = torch.topk(out, 25)
    probs = torch.nn.functional.softmax(probs, 1)

    pred = out.max(1, keepdim=True)[1]

    if float(probs[0,0]) < 0.4:
        predicted_text = 'Sign not detected'
    else:
        percent_prob = float(probs[0, 0]) * 100
        sign = str(int(pred))
        predicted_text = signs[sign] + ': ' + '{:.2f}'.format(percent_prob) + '%'

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    frame = cv2.putText(frame, predicted_text, (60,285), font, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Place your hand in the box", (51, 48), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.rectangle(frame, (50, 80), (250, 250), (0, 255, 0), 3)

    cv2.imshow('Sign Language Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quiting...")
        break
cap.release()

