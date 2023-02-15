import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(1)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    #cv2.imshow('frame', frame)
    # Convert to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
				cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
			param2 = 30, minRadius = 1, maxRadius = 40)

    if detected_circles is not None:

        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a,b,r = pt[0], pt[1], pt[2]
            print("Tọa độ gốc :", a, b)

            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", frame)
            cv2.waitKey(0)
	       
	       
		    

	   
        
cv2.destroyAllWindows()

	    
   
	  







