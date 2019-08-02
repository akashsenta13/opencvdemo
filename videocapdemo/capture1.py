import cv2, time

first_frame = None

cap = cv2.VideoCapture(0)

while (True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if (first_frame is None):
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    threshold_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_delta = cv2.dilate(threshold_delta, None, iterations=2)

    cnts, _ = cv2.findContours(threshold_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for counter in cnts:
        if cv2.contourArea(counter) < 1000:
            continue
            
        # detect object and make rectange around object
        (x, y, w, h) = cv2.boundingRect(counter)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('frame', gray)
    cv2.imshow('delta frame', delta_frame)
    cv2.imshow('threshold frame', threshold_delta)
    cv2.imshow('color frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()