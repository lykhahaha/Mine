import numpy as np
import cv2

# Initialize our canvas as a 300x300 with 3 channels, Red, Green, and Blue, with a black background
canvas = np.zeros((300, 300, 3), dtype='uint8')

# Draw a green line from the top-left corner of our canvas to the bottom-right
green = (0, 255, 0)
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Now, draw a 3 pixel thick red line from the top-right corner to the bottom-left
red = (0, 0, 255)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Draw a green 50x50 pixel square, starting at 10x10 and ending at 60x60
cv2.rectangle(canvas, (10, 10), (60, 60), green)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Draw another rectangle, this time we'll make it red and 5 pixels thick
cv2.rectangle(canvas, (50, 200), (200, 255), red, 5)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Let's draw one last rectangle: blue and filled in
blue = (255, 0, 0)
cv2.rectangle(canvas, (200, 50), (255, 125), blue, -1)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Reset our canvas and draw a white circle at the center of the canvas with increasing radii - from 25 pixels to 150 pixels
canvas = np.zeros((300, 300, 3), dtype='uint8')
centerX, centerY = canvas.shape[1]//2, canvas.shape[0]//2
white = (255, 255, 255)

for r in range(0, 175, 25):
    cv2.circle(canvas, (centerX, centerY), r, white)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Let's go crazy and draw 25 random circles
for i in range(0, 25):
    # randomly generate a radius size between 5 and 200, generate a random color, and then pick a random point on our canvas where the circle will be drawn
    radius = np.random.randint(5, 200)
    color = np.random.randint(0, 256, size=3).tolist()
    pt = np.random.randint(0, 300, 2)

    cv2.circle(canvas, tuple(pt), radius, color, -1)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# exercise
red, black, green = (0, 0, 255), (0, 0, 0), (0, 255, 0)
centerX, centerY = canvas.shape[1]//2, canvas.shape[0]//2
canvas = np.zeros((300, 300, 3), dtype='uint8')

for i in range(0, 300, 20):
    for j in range(0, 300, 20):
        cv2.rectangle(canvas, (i, j), (i+10, j+10), black, -1)
        cv2.rectangle(canvas, (i+10, j), (i+20, j+10), red, -1)

        cv2.rectangle(canvas, (i, j+10), (i+10, j+20), red, -1)
        cv2.rectangle(canvas, (i+10, j+10), (i+20, j+20), black, -1)
cv2.circle(canvas, (centerX, centerY), 50, green, -1)

cv2.imshow('exercise', canvas)
cv2.waitKey(0)