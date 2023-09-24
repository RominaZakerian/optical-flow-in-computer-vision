import numpy as np
import cv2
from numpy import mgrid,vstack
cap = cv2.VideoCapture(0)

def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)


    return lines


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

######################################################3
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
########################################################


fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#creating an object of videoWriter for writing video
out = cv2.VideoWriter('q5_video.avi',fourcc, 5, (int(cap.get(3)),int(cap.get(4))))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#creating an object of videoWriter for writing video
out1 = cv2.VideoWriter('q5_grid_video.avi',fourcc, 5, (int(cap.get(3)),int(cap.get(4))))

mask = np.zeros_like(frame)
# Sets image saturation to maximum
mask[..., 1] = 255
while (1):

    ret, frame = cap.read()
    if ret:

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


        # calculate optical flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0, None,**lk_params)
        flow = cv2.calcOpticalFlowFarneback(old_gray, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)


        lines=draw_flow(frame,flow)
        # create image and draw
        # vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)
        # plot the flow vectors
        # cv2.imshow('Optical flow', frame)
        #
        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break

    # Updating Previous frame and points
        old_gray = image2


        ######################################

        old_gray = image2

        # calculate optical flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0, None,**lk_params)
        flow = cv2.calcOpticalFlowFarneback(old_gray, image3, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        lines = draw_flow(frame, flow)
        # create image and draw
        # vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.circle(frame, (x1, y1), 1, (0, 255, 255), -1)
        # plot the flow vectors
        out1.write(frame)
        out.write(rgb)
        # cv2.imshow('Optical flow', frame)
        cv2.imshow("dense optical flow", rgb)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        # Updating Previous frame and points
        old_gray = image3


cv2.destroyAllWindows()
out.release()
cap.release()

