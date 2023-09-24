import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# params for corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

frames_3=np.zeros((3,480,640))

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
# print(frames_3.shape)

frames_3 = np.delete(frames_3, 0, axis=0)
print(frames_3)
print(frames_3.shape)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
# print(frames_3.shape)

frames_3 = np.delete(frames_3, 0, axis=0)
print(frames_3)
print(frames_3.shape)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
# print(frames_3.shape)

frames_3 = np.delete(frames_3, 0, axis=0)
print(frames_3)
print(frames_3.shape)

frames_3[0] = cv2.normalize(frames_3[0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
frames_3[1] = cv2.normalize(frames_3[1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
frames_3[2] = cv2.normalize(frames_3[2], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
#
# p0 = cv2.goodFeaturesToTrack(old_gray, mask=None,**feature_params)

# Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#creating an object of videoWriter for writing video
out = cv2.VideoWriter('q2_video.avi',fourcc, 15, (int(cap.get(3)),int(cap.get(4))))

while (1):

    ret, frame = cap.read()
    if ret:
        # frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        frames_3 = np.insert(frames_3, frames_3.shape[0], gray, axis=0)
        # print(frames_3.shape)

        frames_3 = np.delete(frames_3, 0, axis=0)
        print(frames_3)
        print(frames_3.shape)

        image1 = cv2.normalize(frames_3[0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        image2 = cv2.normalize(frames_3[1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        image3 = cv2.normalize(frames_3[2], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        old_gray = image1

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,image2,p0, None,**lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        mask = np.zeros_like(frame)
        for i, (new, old) in enumerate(zip(good_new,good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),color=(0,255,255), thickness=2)

            frame = cv2.circle(frame, (int(a), int(b)), 5,color=(0,255,255), thickness=-1)

        img = cv2.add(frame, mask)

        # cv2.imshow('frame', img)
        #
        # k = cv2.waitKey(25)
        # if k == 27:
        #     break

        # Updating Previous frame and points
        old_gray = image2.copy()
        p0 = good_new.reshape(-1, 1, 2)

    #######################################

        old_gray_2 = image2

        p0_2 = cv2.goodFeaturesToTrack(old_gray_2, mask=None, **feature_params)

        p1_2, st_2, err = cv2.calcOpticalFlowPyrLK(old_gray_2,image3,p0_2, None,**lk_params)

        # Select good points
        good_new_2 = p1_2[st_2 == 1]
        good_old_2 = p0_2[st_2 == 1]

        # draw the tracks
        mask_2 = np.zeros_like(frame)
        for i, (new, old) in enumerate(zip(good_new_2,good_old_2)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask_2, (int(a), int(b)), (int(c), int(d)),color=(0,0,255), thickness=2)

            frame = cv2.circle(frame, (int(a), int(b)), 5,color=(0,0,255), thickness=-1)

        img = cv2.add(frame, mask)

        out.write(img)
        cv2.imshow('frame', img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


        # Updating Previous frame and points
        old_gray_2 = image3.copy()
        p0_2 = good_new_2.reshape(-1, 1, 2)
cv2.destroyAllWindows()
cap.release()
out.release()