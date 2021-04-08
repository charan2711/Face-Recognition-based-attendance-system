#IMPORTING LIBRARIES
import cv2
import numpy as np
import face_recognition

#IMPORTING IMAGES
imag1 = face_recognition.load_image_file('Imagebasics/rakesh.jpeg')
imag1 = cv2.cvtColor(imag1,cv2.COLOR_BGR2RGB)

test1 = face_recognition.load_image_file('Imagebasics/rakesh test.jpeg')
test1 = cv2.cvtColor(test1,cv2.COLOR_BGR2RGB)

#Finding the face location  and encoding the images
faceLoc = face_recognition.face_locations(imag1)[0]
encodeimag = face_recognition.face_encodings(imag1)[0]
cv2.rectangle(imag1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(test1)[0]
encodeTest = face_recognition.face_encodings(test1)[0]
cv2.rectangle(test1,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Cheacking if the face is matched with the test face
results = face_recognition.compare_faces([encodeimag],encodeTest)
distance = face_recognition.face_distance([encodeimag],encodeTest)
print(results,distance)

#Adding text on my test image
cv2.putText(test1,f'{results} {round(distance[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)


#Displaying the images
cv2.imshow('Rakesh',imag1)
cv2.imshow('rakesh test image',test1)
cv2.waitKey(0)