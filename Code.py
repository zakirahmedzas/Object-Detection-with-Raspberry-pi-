#https://github.com/zakirahmedzas
# DROBAN
import cv2
import logging
import time
import os

# Function to create unique filenames based on timestamp
def generate_unique_filename(base_path, extension):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_path}_{timestamp}{extension}"

# Load class names
classNames = []
classFile = "/home/zakir/Desktop/Object_Detection_Files/coco.names" #Replace this path
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load the pre-trained model
configPath = "/home/zakir/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" # Replace this path
weightsPath = "/home/zakir/Desktop/Object_Detection_Files/frozen_inference_graph.pb" # Replace this path

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to get objects
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Log detection details
                logging.info(f"Detected: {className}, Confidence: {round(confidence * 100, 2)}%, Box: {box}")
    return img, objectInfo


if __name__ == "__main__":

    # Create unique filenames for video and log
    video_filename = generate_unique_filename('/home/zakir/Desktop/Video and Log Data/Video', '.avi')
    log_filename = generate_unique_filename('/home/zakir/Desktop/Video and Log Data/Log', '.txt')

    # Setup logging for this session (writing to unique log file)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height

    # Initialize VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
    out = cv2.VideoWriter(video_filename, fourcc, 4.0, (640, 480))  # Output file, codec, FPS, frame size

    while True:
        success, img = cap.read()
        if not success:
            break

        # Perform object detection
        result, objectInfo = getObjects(img, 0.45, 0.2)

        # Show the resulting frame with detections
        cv2.imshow("Output", img)

        # Write the frame with detections to the output video file
        out.write(img)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
