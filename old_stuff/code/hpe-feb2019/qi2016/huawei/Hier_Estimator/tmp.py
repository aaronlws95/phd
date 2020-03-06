import cv2
import numpy as np
import numpy


M = cv2.getRotationMatrix2D((48,48),20,1.2)
offset = numpy.dot(M,numpy.array([80,10,1]))
M = cv2.getRotationMatrix2D((48,48),-20,1/1.2)
offset = numpy.dot(M,numpy.array([offset[0],offset[1],1]))
print(offset)
