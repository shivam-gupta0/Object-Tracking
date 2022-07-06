import cv2
import numpy as np
import math



cfg_path = 'E:\\project\\vehicle_dataset\\model_test\\yolov3_custom1.cfg'
weights_path = "E:\\project\\vehicle_dataset\\model_test\\yolov3_custom1.weights"

def detection(img, cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3] * ht)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


classes = []

with open("E:\\project\\vehicle_dataset\\new_images\\test_data//classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

old_center_p = []
id_dict = {}
ids = 0
cap = cv2.VideoCapture("E:\\project\\vehicle_dataset\\model_test\\3.webm")

while cap.isOpened():
    rate, frame = cap.read()
    scale_percent = 70  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    my_img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    ht, wt, _ = my_img.shape

    boxes, confidences, class_ids = detection(my_img, cfg_path, weights_path)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    center_p = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_p.append((cx, cy))
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(my_img, label + " " + confidence, (x, y + 10), font, 1, (255, 255, 0), 1)

        if ids == 0:
            for p in center_p:
                id_dict[ids] = p
                ids += 1
        id_dict_copy = id_dict.copy()
        center_p_copy = center_p.copy()
        if ids > 0:
            for idn, pt in id_dict_copy.items():
                point_ext = False
                for pt1 in center_p_copy:
                    dt = math.hypot(pt[0] - pt1[0], pt[1] - pt1[1])
                    if dt <= 20:
                        id_dict[idn] = pt1
                        center_p.remove(pt1)
                        point_ext = True
                if not point_ext:
                    id_dict.pop(idn)
            for pt2 in center_p:
                id_dict[ids] = pt2
                ids += 1

        for idn, pt in id_dict.items():
            cv2.circle(my_img, pt, 2, (0, 0, 255), -1)
            cv2.putText(my_img, str(idn), (pt[0], pt[1] - 7), font, 1, (0, 0, 255), 1)
        print("previous", "\n", old_center_p, "\n", "current", "\n", center_p, "\n", "id_dictionary", id_dict)
        old_center_p = center_p_copy.copy()

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('image', 1200, 600)
        cv2.imshow('image', my_img)

        if cv2.waitKey(1) == ord('q'):
            continue
        elif cv2.waitKey(1) == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()
