# load HL detection model from imageAI
# open camera with openCV, analyze frame by frame
# draw a red frame around the detected object
# display FPS

from imageai.Detection.Custom import CustomObjectDetection
import os
import cv2
import time

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hololens-ex-60--loss-2.76.h5") 
detector.setJsonPath("detection_config.json")
detector.loadModel()

# init camera
execution_path = os.getcwd()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while True:
    # FPS process
    start_time = time.time()

    # Grab a single frame of video
    ret, frame = camera.read()
    fast_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    detected_image, detections = detector.detectObjectsFromImage(input_image=fast_frame, input_type="array", output_type="array")

    for detection in detections:
        # frame for the detected object
        (x1, y1, x2, y2) = detection["box_points"]
        x1 *= 4
        y1 *= 4
        x2 *= 4
        y2 *= 4

        (x1, y1, x2, y2) = detection["box_points"]
        detectionInfo = detection["name"] + " : " + str(detection["percentage_probability"]) + "x1: " + str(x1) + " - y1: " + str(y1) + " - x2: "+ str(x2) + " - y2: " + str(y2)
        print(detectionInfo)

    #display FPS
    fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    print(fpsInfo)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

