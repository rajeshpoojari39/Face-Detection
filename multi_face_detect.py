import cv2, glob

gimp = glob.glob("*.jpg")
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for timg in gimp:
    img = cv2.imread(timg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Multi Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
