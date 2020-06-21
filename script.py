# importing needed modules-----------------------
import cv2, torch
import numpy as np
from FERModel import *

# defining important funtions---------------------

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# function to turn photos to tensor
def img2tensor(x):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return transform(x)

# the model for predicting
model = FERModel(1, 7)
softmax = torch.nn.Softmax(dim=1)
model.load_state_dict(torch.load('FER2013-Resnet9.pth', map_location=get_default_device()))

def predict(x):
    out = model(img2tensor(img)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    return {'label': label, 'probability': prob}

if __name__ == "__main__":
    vid = cv2.VideoCapture(0)

    while vid.isOpened():
        _, frame = vid.read()

        # takes in a gray coloured filter of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # initializing the haarcascade face detector
        faces = face_cascade.detectMultiScale(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # takes the region of interest of the face only in gray
            roi_gray = gray[y:y+h, x:x+h]
            resized = cv2.resize(roi_gray, (48, 48))    # resizes to 48x48 sized image

            # predict the mood
            img = img2tensor(resized)
            prediction = predict(img)

            cv2.putText(frame, f"{prediction['label']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))


        cv2.imshow('video', frame)
        if cv2.waitKey(1) == 27:
            break

    vid.release()
    cv2.destroyAllWindows()