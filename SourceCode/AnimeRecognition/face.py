import cv2
import sys


'''
Definitions
   Cascade: OpenCV looks through certain chunks of a picture detecting whether it is a face, if it is it begins doing a more complex search.  IF nah then it moves on
'''

'''The User Supplied Variables '''


CascPath = "lbpcascade_animeface.xml"

'''Creating the cascade and function'''
#Default Cascade
def Detect(ImagePath):
    FaceCascade = cv2.CascadeClassifier(CascPath)

    '''Reading The Image and Converting it to grayscale '''

    image = cv2.imread(ImagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    ''' General Function that detects the objects '''
    ''' Selects the Grayscale image'''
    ''' is the scaleFactor. Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this'''
    ''' detection algorithm uses a moving window to detect objects. minNeighbors defines how many objects are detected near the current one before it declares the face found. minSize, meanwhile, gives the size of each window.'''

    faces = FaceCascade.detectMultiScale(
        gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=(30,30),
    )

    print("Found {0} faces".format(len(faces)))

    ''' Drawing a rectagnel around the faces '''

    for (x, y , w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
    cv2.imwrite("out.png", image)

    if len(sys.argv) != 2:
        sys.stderr.write("usage: detect.py <filename>\n")
        sys.exit(-1)

    
Detect(sys.argv[1])