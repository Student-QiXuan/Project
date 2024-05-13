import cv2
import numpy as np
import imutils
import RPi.GPIO as GPIO
from time import sleep
import os
import VideoStream
import time

## Camera settings
IM_WIDTH = 320
IM_HEIGHT = 240 
FRAME_RATE = 10

videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up


flag=0

class PID:
    def __init__(self, Kp, Ki, Kd, min_output, max_output):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_output = min_output
        self.max_output = max_output
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # Skip integral term calculation if Ki is zero
        if self.Ki != 0:
            self.integral += error

            # Anti-windup (limiting integral term)
            self.integral = max(self.min_output / self.Ki, min(self.max_output / self.Ki, self.integral))

        derivative = error - self.prev_error
        self.prev_error = error

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        # Constrain output within range of min_output to max_output
        output = max(self.min_output, min(self.max_output, output))
        return output

in1 = 17
in2 = 27
in3 = 22
in4 = 23

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

# Create a PWM object for controlling the speed
# Set up hardware PWM
pwm1 = GPIO.PWM(in1, 100)  # Pin, Frequency
pwm2 = GPIO.PWM(in2, 100)  # Pin, Frequency
pwm3 = GPIO.PWM(in3, 100)  # Pin, Frequency
pwm4 = GPIO.PWM(in4, 100)  # Pin, Frequency

#Color threshold
lower_red = np.array([161,107,0])
upper_red = np.array([179,255,255])
     
lower_green = np.array([75,165,61])
upper_green = np.array([79,255,99])

lower_yellow = np.array([20,100,76])
upper_yellow = np.array([33,255,255])

lower_blue = np.array([81,70,60])
upper_blue = np.array([130,255,255])

lower_black = np.array([0,0,0])
upper_black = np.array([179,255,54])

img1= "Train/Stop"
img2= "Train/Traffic Light"
img3= "Train/Face"
img4= "Train/distance"

stop_template = cv2.imread(img1+".png", 0)
traffic_template = cv2.imread(img2+".png",0)
face_template = cv2.imread(img3+".png",0)
distance_template = cv2.imread(img4+".png",0)

resized_stop = cv2.resize(stop_template, (320, 240))
resized_traffic = cv2.resize(traffic_template, (320, 240))
resized_face = cv2.resize(face_template, (320, 240))
resized_distance = cv2.resize(distance_template, (320, 240))


    # Template matching function
def match_template(img, template):
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val
    # Function to identify arrow direction
def identify_symbol(img_gray):
    stop_score = match_template(img_gray, resized_stop)
    traffic_score = match_template(img_gray, resized_traffic)
    face_score = match_template(img_gray, resized_face)
    distance_score = match_template(img_gray, resized_distance)
    threshold_score = 0.35  # Adjust as needed
    print("Stop score:", stop_score)
    print("Traffic score:", traffic_score)
    print("Face score:", face_score)
    print("Distance score:", distance_score)
    if stop_score > threshold_score and stop_score > traffic_score and stop_score > face_score and stop_score > distance_score:
        return "Stop"
    elif traffic_score > threshold_score and traffic_score > stop_score and traffic_score > face_score and traffic_score > distance_score:
        return "Traffic"
    elif face_score > threshold_score and face_score > stop_score and face_score > traffic_score  and face_score > distance_score:
        return "Face"
    elif distance_score > threshold_score and distance_score > stop_score and distance_score > face_score and distance_score > traffic_score:
        return "Distance"
    else: return ""

def shapeDetection(image):
    cv2.imshow("image",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #cv2.imshow("blur", blur)
    # Perform edge detection
    edges = cv2.Canny(blur, 50, 150)
    #cv2.imshow("edges", edges)
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Classifying shapes
    for contour in contours:
        #cv2.imshow("contours", contour)
        epsilon = 0.03 * cv2.arcLength(contour, True)
        #print("epsilon:", epsilon)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        #print("Vertices:", vertices)
        area = cv2.contourArea(contour)
        #cv2.imshow("area", area)
        
        if area > 900:
            if vertices == 3:
                shape = "Triangle"
                print(shape)
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                return shape
            elif vertices == 4:
                shape = "Rectangle"
                print(shape)
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                return shape
            elif vertices == 5:
                shape = "Pentagon"
                print(shape)
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                return shape
            elif vertices == 6:
                shape = "Hexagon"
                print(shape)
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
            elif vertices == 8:
                
                (x, y), (w, h), angle = cv2.fitEllipse(contour)  # Fit ellipse to contour
                circularity = cv2.contourArea(contour) / (np.pi * (w / 2) * (h / 2))  # Calculate circularity
                #print(circularity)
                if circularity < 0.99:  # Threshold for circularity
                    shape = "Partial Circle"
                    print(shape)
                    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                    cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    return shape
                else:
                    shape = "Circle"
                    print(shape)
                    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                    cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    return shape
                      
            elif vertices == 7:
                shape = ""
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Calculate angle between centroid and tip of the arrow
                    angle = np.arctan2(approx[0][0][1] - cY, approx[0][0][0] - cX) * 180 / np.pi
                    print(angle)
                    
                    if angle < 0:
                        angle += 360
                        # Determine direction based on angle
                    if 0 < angle <= 45 or 315 < angle <= 360:
                        direction = "Right"
                        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                        cv2.putText(image, direction, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    elif 135 < angle <= 225:
                        direction = "Left"
                        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                        cv2.putText(image, direction, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    elif 225 < angle <= 315:
                        direction = "Down"
                        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                        cv2.putText(image, direction, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    else:
                        direction = "Up"
                        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                        cv2.putText(image, direction, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                shape = direction
                # Put text at the centroid
                #print(shape)
                return shape
                flag=1
                return flag
                #cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
                #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
      
min_output = -80  # Minimum PWM value
max_output = 80   # Maximum PWM value
# Parameters for the PID controller
Kp = 0.6
Ki = 0.0
Kd = 0.0

pid = PID(Kp, Ki, Kd, min_output, max_output)


while True:
    frame= videostream.read()
    frame=frame[130:240,0:320]
    #cv2.imshow("Frame",frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame1 = videostream.read()
    shapeDetection(frame1)
    symbol = identify_symbol(gray_frame)
    print(symbol)
    # Perform color thresholding for each color
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours for each color mask
    red_contours = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours= cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours= cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours= cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_contours= cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    red_contours = imutils.grab_contours(red_contours)
    yellow_contours = imutils.grab_contours(yellow_contours)
    blue_contours = imutils.grab_contours(blue_contours)
    green_contours = imutils.grab_contours(green_contours)
    black_contours = imutils.grab_contours(black_contours)
    
    if len(green_contours)>0:
        c = max(green_contours, key=cv2.contourArea)
        cv2.drawContours(frame, c, -1, (0,255,0), 1)
        M = cv2.moments(c)
        if M["m00"] !=0 :
           cx = int(M['m10']/M['m00'])
           #cy = int(M['m01']/M['m00'])
           error = cx - 160  # Calculate error (difference between centroid and center of the frame)

           # Use PID controller to adjust motor speeds based on the error
           correction = pid.update(error)
           #print(correction)
            
           left_speed = 20 + correction
           right_speed = 20 - correction
           Tleft_speed = 25 + correction
           Tright_speed = 25 - correction
           
           uplimit = 55
           dolimit = 0
           Tuplimit = 80
           Tdolimit = 0
           
           if left_speed > uplimit:
               left_speed = uplimit
           elif left_speed < dolimit:
               left_speed = dolimit
           if right_speed > uplimit:
               right_speed = uplimit
           elif right_speed < dolimit:
               right_speed = dolimit
           if Tleft_speed > Tuplimit:
               Tleft_speed = Tuplimit
           elif Tleft_speed < Tdolimit:
               Tleft_speed = Tdolimit
           if Tright_speed > Tuplimit:
               Tright_speed = Tuplimit
           elif Tright_speed < Tdolimit:
               Tright_speed = Tdolimit
               
           if cx >= 283 :
                #print("Turn Right")
                pwm1.start(Tleft_speed)
                pwm2.start(0)
                pwm3.start(0)
                pwm4.start(left_speed) 
           elif cx < 283 and cx >37 :
                #print("On Track!")
                pwm1.start(left_speed)
                pwm2.start(0)
                pwm3.start(right_speed)
                pwm4.start(0)
           elif cx <=37:
                #print("Turn Left")
                pwm1.start(0)
                pwm2.start(right_speed)
                pwm3.start(Tright_speed)
                pwm4.start(0)
                
    elif len(red_contours)>0:
        c = max(red_contours, key=cv2.contourArea)
        cv2.drawContours(frame, c, -1, (0,255,0), 1)
        M = cv2.moments(c)
        if M["m00"] !=0 :
           cx = int(M['m10']/M['m00'])
           #cy = int(M['m01']/M['m00'])         
           
           error = cx - 160  # Calculate error (difference between centroid and center of the frame)

           # Use PID controller to adjust motor speeds based on the error
           correction = pid.update(error)
           #print(correction)
            
           left_speed = 20 + correction
           right_speed = 20 - correction
           Tleft_speed = 25 + correction
           Tright_speed = 25 - correction
           
           uplimit = 55
           dolimit = 0
           Tuplimit = 80
           Tdolimit = 0
           
           if left_speed > uplimit:
               left_speed = uplimit
           elif left_speed < dolimit:
               left_speed = dolimit
           if right_speed > uplimit:
               right_speed = uplimit
           elif right_speed < dolimit:
               right_speed = dolimit
           if Tleft_speed > Tuplimit:
               Tleft_speed = Tuplimit
           elif Tleft_speed < Tdolimit:
               Tleft_speed = Tdolimit
           if Tright_speed > Tuplimit:
               Tright_speed = Tuplimit
           elif Tright_speed < Tdolimit:
               Tright_speed = Tdolimit
               
           if cx >= 283 :
                #print("Turn Right")
                pwm1.start(Tleft_speed)
                pwm2.start(0)
                pwm3.start(0)
                pwm4.start(left_speed) 
           elif cx < 283 and cx >37 :
                #print("On Track!")
                pwm1.start(left_speed)
                pwm2.start(0)
                pwm3.start(right_speed)
                pwm4.start(0)
           elif cx <=37:
                #print("Turn Left")
                pwm1.start(0)
                pwm2.start(right_speed)
                pwm3.start(Tright_speed)
                pwm4.start(0)
                
    elif len(yellow_contours)>0:
        c = max(yellow_contours, key=cv2.contourArea)
        cv2.drawContours(frame, c, -1, (0,255,0), 1)
        M = cv2.moments(c)
        if M["m00"] !=0 :
           cx = int(M['m10']/M['m00'])
           #cy = int(M['m01']/M['m00'])
           
           error = cx - 160  # Calculate error (difference between centroid and center of the frame)

           # Use PID controller to adjust motor speeds based on the error
           correction = pid.update(error)
           #print(correction)
            
           left_speed = 20 + correction
           right_speed = 20 - correction
           Tleft_speed = 25 + correction
           Tright_speed = 25 - correction
           
           uplimit = 55
           dolimit = 0
           Tuplimit = 80
           Tdolimit = 0
           
           if left_speed > uplimit:
               left_speed = uplimit
           elif left_speed < dolimit:
               left_speed = dolimit
           if right_speed > uplimit:
               right_speed = uplimit
           elif right_speed < dolimit:
               right_speed = dolimit
           if Tleft_speed > Tuplimit:
               Tleft_speed = Tuplimit
           elif Tleft_speed < Tdolimit:
               Tleft_speed = Tdolimit
           if Tright_speed > Tuplimit:
               Tright_speed = Tuplimit
           elif Tright_speed < Tdolimit:
               Tright_speed = Tdolimit
               
           if cx >= 283 :
                #print("Turn Right")
                pwm1.start(Tleft_speed)
                pwm2.start(0)
                pwm3.start(0)
                pwm4.start(left_speed) 
           elif cx < 283 and cx >37 :
                #print("On Track!")
                pwm1.start(left_speed)
                pwm2.start(0)
                pwm3.start(right_speed)
                pwm4.start(0)
           elif cx <=37:
                #print("Turn Left")
                pwm1.start(0)
                pwm2.start(right_speed)
                pwm3.start(Tright_speed)
                pwm4.start(0)
    
    elif len(blue_contours)>0:
        c = max(blue_contours, key=cv2.contourArea)
        cv2.drawContours(frame, c, -1, (0,255,0), 1)
        M = cv2.moments(c)
        if M["m00"] !=0 :
           cx = int(M['m10']/M['m00'])
           #cy = int(M['m01']/M['m00'])
           
           error = cx - 160  # Calculate error (difference between centroid and center of the frame)

           # Use PID controller to adjust motor speeds based on the error
           correction = pid.update(error)
           #print(correction)
            
           left_speed = 20 + correction
           right_speed = 20 - correction
           Tleft_speed = 25 + correction
           Tright_speed = 25 - correction
           
           uplimit = 55
           dolimit = 0
           Tuplimit = 80
           Tdolimit = 0
           
           if left_speed > uplimit:
               left_speed = uplimit
           elif left_speed < dolimit:
               left_speed = dolimit
           if right_speed > uplimit:
               right_speed = uplimit
           elif right_speed < dolimit:
               right_speed = dolimit
           if Tleft_speed > Tuplimit:
               Tleft_speed = Tuplimit
           elif Tleft_speed < Tdolimit:
               Tleft_speed = Tdolimit
           if Tright_speed > Tuplimit:
               Tright_speed = Tuplimit
           elif Tright_speed < Tdolimit:
               Tright_speed = Tdolimit
               
           if cx >= 283 :
                #print("Turn Right")
                pwm1.start(Tleft_speed)
                pwm2.start(0)
                pwm3.start(0)
                pwm4.start(left_speed) 
           elif cx < 283 and cx >37 :
                #print("On Track!")
                pwm1.start(left_speed)
                pwm2.start(0)
                pwm3.start(right_speed)
                pwm4.start(0)
           elif cx <=37:
                #print("Turn Left")
                pwm1.start(0)
                pwm2.start(right_speed)
                pwm3.start(Tright_speed)
                pwm4.start(0)
                
    elif len(black_contours)>0:
        #print('red')
        c = max(black_contours, key=cv2.contourArea)
        cv2.drawContours(frame, c, -1, (0,255,0), 1)
        M = cv2.moments(c)
        if M["m00"] !=0 :
           cx = int(M['m10']/M['m00'])
           #cy = int(M['m01']/M['m00'])
           error = cx - 160  # Calculate error (difference between centroid and center of the frame)

           # Use PID controller to adjust motor speeds based on the error
           correction = pid.update(error)
           #print(correction)
            
           left_speed = 20 + correction
           right_speed = 20 - correction
           Tleft_speed = 25 + correction
           Tright_speed = 25 - correction
           
           uplimit = 55
           dolimit = 0
           Tuplimit = 80
           Tdolimit = 0
           
           if left_speed > uplimit:
               left_speed = uplimit
           elif left_speed < dolimit:
               left_speed = dolimit
           if right_speed > uplimit:
               right_speed = uplimit
           elif right_speed < dolimit:
               right_speed = dolimit
           if Tleft_speed > Tuplimit:
               Tleft_speed = Tuplimit
           elif Tleft_speed < Tdolimit:
               Tleft_speed = Tdolimit
           if Tright_speed > Tuplimit:
               Tright_speed = Tuplimit
           elif Tright_speed < Tdolimit:
               Tright_speed = Tdolimit
               
           if cx >= 283 :
                #print("Turn Right")
                pwm1.start(Tleft_speed)
                pwm2.start(0)
                pwm3.start(0)
                pwm4.start(left_speed) 
           elif cx < 283 and cx >37 :
                #print("On Track!")
                pwm1.start(left_speed)
                pwm2.start(0)
                pwm3.start(right_speed)
                pwm4.start(0)
           elif cx <=37:
                #print("Turn Left")
                pwm1.start(0)
                pwm2.start(right_speed)
                pwm3.start(Tright_speed)
                pwm4.start(0)
           else :
                print("I don't see the line")
                pwm1.start(0)
                pwm2.start(0)
                pwm3.start(0)
                pwm4.start(0)
                
    #else:         
        #print("I dont see the line")
        #pwm1.start(0)
        #pwm2.start(25)
        #pwm3.start(0)
        #pwm4.start(25)
# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()


