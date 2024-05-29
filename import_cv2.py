import cv2
import time
import numpy as np



def main(): 
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Initialize video capture (you can also use a video file by replacing 0 with the file path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        # Create a 4D blob from a frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Processing detections
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green color for bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with detections
        cv2.imshow("YOLO Object Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# def main():
#     # Load the pre-trained Haar Cascade classifier for full body detection
#     person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

#     # Capture video from the default camera
#     cap = cv2.VideoCapture(0)

#     # Store the detected persons' positions with timestamps
#     detected_persons = []

#     while True:
#         # Read a frame from the video capture
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect persons in the grayscale frame
#         persons = person_cascade.detectMultiScale(gray, 1.1, 4)

#         # Store detected persons' positions with timestamps
#         for (x, y, w, h) in persons:
#             detected_persons.append((x, y, w, h, time.time()))
#             print("Person detected at coordinates: x={}, y={}, width={}, height={}".format(x, y, w, h))

#         # Draw rectangles around detected persons
#         current_time = time.time()
#         for (x, y, w, h, timestamp) in detected_persons:
#             # Keep the frame for 5 seconds
#             if current_time - timestamp < 5:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Display the frame with detections
#         cv2.imshow('Person Detection', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Remove old detections
#         detected_persons = [(x, y, w, h, timestamp) for (x, y, w, h, timestamp) in detected_persons if current_time - timestamp < 5]

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

    # person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # cap = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     persons = person_cascade.detectMultiScale(gray, 1.1, 4)
        
    #     for (x, y, w, h) in persons:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #         print("Alert! person detected.")

    #     cv2.imshow('Person Detection', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

# def get_hsv_range():
#     print("Enter the HSV color range for detection.")
#     h_min = int(input("Enter H minimum value (0-179): "))
#     h_max = int(input("Enter H maximum value (0-179): "))
#     s_min = int(input("Enter S minimum value (0-255): "))
#     s_max = int(input("Enter S maximum value (0-255): "))
#     v_min = int(input("Enter V minimum value (0-255): "))
#     v_max = int(input("Enter V maximum value (0-255): "))
    
#     lower_range = np.array([h_min, s_min, v_min])
#     upper_range = np.array([h_max, s_max, v_max])
    
#     return lower_range, upper_range

# def detect_color(frame, lower_range, upper_range):
#     # Convert the frame to the HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Threshold the HSV image to get only the desired colors
#     mask = cv2.inRange(hsv, lower_range, upper_range)
    
#     # Apply some morphological operations to remove noise
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)
    
#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     return contours, mask

# def main():
#     lower_range, upper_range = get_hsv_range()
    
#     # Initialize the video capture
#     cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
        
#         if not ret:
#             break
        
#         # Detect the specified color in the current frame
#         contours, mask = detect_color(frame, lower_range, upper_range)
        
#         # Alert the user if the specified color is detected
#         if contours:
#             print("Alert! Object detected.")
#             for contour in contours:
#                 if cv2.contourArea(contour) > 500:
#                     (x, y, w, h) = cv2.boundingRect(contour)
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Display the resulting frame
#         cv2.imshow('Detection', frame)
#         cv2.imshow('Mask', mask)
        
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release the video capture and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
