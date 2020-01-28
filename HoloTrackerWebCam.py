# load HL detection model from imageAI
# open camera with openCV, analyze frame by frame
# draw a red frame around the detected object
# display FPS, resize image to 1/4 to improve performance

from djitellopy import Tello
from imageai.Detection.Custom import CustomObjectDetection
import os
import cv2
import time

######################################################################
width = 320  # WIDTH OF THE IMAGE
height = 240  # HEIGHT OF THE IMAGE
startCounter =0   #  0 FOR FIGHT 1 FOR TESTING
######################################################################

# load hl model detector
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hololens-ex-60--loss-2.76.h5") 
detector.setJsonPath("detection_config.json")
detector.loadModel()

# CONNECT TO TELLO
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

print(me.get_battery())

me.streamoff()
me.streamon()

while True:
    # FPS process
    start_time = time.time()

    # Grab a single frame of video
    # GET THE IMGAE FROM TELLO
    frame_read = me.get_frame_read()
    frame = frame_read.frame
    img = cv2.resize(frame, (width, height))

    fast_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    detected_image, detections = detector.detectObjectsFromImage(input_image=fast_frame, input_type="array", output_type="array")

    for detection in detections:
        # frame for the detected object
        (x1, y1, x2, y2) = detection["box_points"]
        x1 *= 4
        y1 *= 4
        x2 *= 4
        y2 *= 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw a label with the detected object type below the frame
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, detection["name"], (x1 + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)

    #display FPS
    fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    print(fpsInfo)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, fpsInfo, (10, 20), font, 0.4, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

