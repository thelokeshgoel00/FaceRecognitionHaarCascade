import numpy as np
import os
import cv2


def dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def knn(train, test, k=5):
    vals = []
    m = train.shape[0]

    for i in range(1, m):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = dist(test, ix)
        vals.append((d, iy))
    vals = sorted(vals, key=lambda x: x[0])[:k]

    vals = np.array(vals)[:, -1]
    new_vals = np.unique(vals, return_counts=True)
    # print(newvals)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haar-cascade-files-master/haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
dataset_path = 'MyData/'

label = []
class_id = 0  # labels for the given file
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face

        offset = 10
        frame = np.array(frame)
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 251), 2)
    cv2.imshow("faces", frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
