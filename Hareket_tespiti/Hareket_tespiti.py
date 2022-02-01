from audioop import mul
import cv2
import numpy as np
#Webcam görüntüsünde tespit edilen hareketi çerçeveleyip kaydeden program

cap = cv2.VideoCapture(0)

#Hareket tespit edilen videoyu "XVID" formatında kaydeder.
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter(r"opencv\video_ve_resimler\output.avi", fourcc, 5.0, (1280,720))

while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    #absdiff:2 frame arası çıkarma işlemi yapar. Mutlak değer alır. Sonuçta değişen kısımlar(hareketli kısım) bulunur.
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:           
            continue  
   
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame1, "Movement", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)


       
    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
out.release()