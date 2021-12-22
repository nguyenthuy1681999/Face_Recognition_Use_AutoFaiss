from mtcnn import MTCNN
import cv2
from PIL import Image
from PIL import Image
from face_search import search_face
import os 

#Resize Image  
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Detect face, search face and display reuslts
detector = MTCNN()
def tagging_image(path): 
    im = Image.open(path)
    image = cv2.imread(path)
    faces = detector.detect_faces(image)
    print("Results:")
    for face in faces:
        if face['confidence'] > 0.8:
            bounding_box = face['box']
            cv2.rectangle(image,(bounding_box[0], bounding_box[1]), 
                        (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), 
                        (0,204,0),2)
            crop_img = im.crop((bounding_box[0], bounding_box[1],
                                bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]))
            name = search_face(crop_img)
            cv2.putText(image,name, (bounding_box[0], bounding_box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    print("---------------")
    img = ResizeWithAspectRatio(image, width=640)
    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)

# TEST
# Test with folder test
image_folder_in = "./data/test/"
for image_name in os.listdir(image_folder_in):
            image_path = os.path.join(image_folder_in,image_name)
            tagging_image(image_path)
        
# Test with one image, please uncomment line under: 
# tagging_image("./data/test/5.jpg")