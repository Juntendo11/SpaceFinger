import cv2
import mediapipe as mp
import time
import math
import mathutils
import numpy as np

# class creation
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 20 landmark points

    def findHands(self,img,draw=False):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the frame
    #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    #Draw dots and connect them
                    self.mpDraw.draw_landmarks(img,handLms,
                                                self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handNo=0, draw=True):
        """Lists the position/type of landmarks
        we give in the list and in the list ww have stored
        type and position of the landmarks.
        List has all the lm position"""
        lmlist = []
        # check wether any landmark was detected
        if self.results.multi_hand_landmarks:
            #Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[handNo]
            # Get id number and landmark information
            for id, lm in enumerate(myHand.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h,w,c = img.shape
                
                #find the position
                lx,ly = int(lm.x*w), int(lm.y*h)    #Center in pixel
                gx, gy, gz = float(lm.x), float(lm.y), float(lm.z) #Local axis z ~ x scale magnitude

                lmlist.append([id,gx,gy,gz,lx,ly])
        return lmlist

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def calibrate(lmlists,samples):
    """
    Calibration using samples
    """
    print("Calibrating")
    gx1_sum, gy1_sum, gz1_sum, gx2_sum, gy2_sum, gz2_sum = 0, 0, 0, 0, 0, 0
    gx1_avg, gy1_avg, gz1_avg, gx2_avg, gy2_avg, gz2_avg = 0, 0, 0, 0, 0, 0
    
    for lmlist in lmlists:
        if len(lmlist) != 0:
            gx1, gy1, gz1 = lmlist[4][1], lmlist[4][2], lmlist[4][3]
            gx2, gy2, gz2 = lmlist[8][1], lmlist[8][2], lmlist[4][3]
            gx1_sum += gx1
            gy1_sum += gy1
            gz1_sum += gz1
            gx2_sum += gx2
            gy2_sum += gy2
            gz2_sum += gz2
            
    #Global average
    gx1_avg = gx1_sum/samples
    gx2_avg = gx2_sum/samples
    gy1_avg = gy1_sum/samples
    gy2_avg = gy2_sum/samples
    gz1_avg = gz1_sum/samples
    gz2_avg = gz2_sum/samples

    #Center average
    cgx_avg, cgy_avg, cgz_avg = (gx1_avg + gx2_avg) / 2, (gy1_avg + gy2_avg) / 2, (gz1_avg + gz2_avg) /2
    
    pos_avg = np.array([cgx_avg, cgy_avg, cgz_avg]) #Pos
    
    dist_avg = math.sqrt((gx2_avg-gx1_avg)**2 + (gy2_avg-gy1_avg)**2 + (gz2_avg-gz1_avg)**2) #Scale
    vec_avg = np.array([gx2_avg - gx1_avg, gy2_avg - gy1_avg, gz2_avg - gz1_avg]) #Rotation
    norm_vec_avg = vec_avg/np.linalg.norm(vec_avg)
    print("Calibrated")
    return pos_avg, dist_avg, vec_avg

def main():
    #Frame rates
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    calibrating = False
    calibrated = False
    samples = 100
    lmlists = []
    
    pos_avg = np.zeros(3)
    dist_avg = 0
    vec_avg = np.zeros(3)
    
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        #Calibration
        if calibrating == True:
            cv2.putText(img,"Calibrating",(300,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            lmlists.append(lmlist)
            if len(lmlists) > samples:
                pos_avg, dist_avg, vec_avg = calibrate(lmlists,samples)
                calibrating = False
                calibrated = True
        elif calibrated == False:
            cv2.putText(img,"C calibrate",(300,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        else:
            cv2.putText(img,"Calibrated",(300,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
                
        if len(lmlist) != 0:
            gx1, gy1, gz1 = lmlist[4][1], lmlist[4][2], lmlist[4][3]
            gx2, gy2, gz2 = lmlist[8][1], lmlist[8][2], lmlist[4][3]
            
            lx1, ly1 = lmlist[4][4],  lmlist[4][5]  #Pixel coordinate
            lx2, ly2 = lmlist[8][4],  lmlist[8][5]
            clx, cly =  (lx1 + lx2) // 2, (ly1 + ly2) // 2 #Center of two points
            cgx, cgy,cgz = (gx1+gx2)/2, (gy1+gy2)/2, (gz1+gz2)/2
            
            cv2.circle(img, (lx1, ly1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (lx2,ly2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (lx1, ly1), (lx2, ly2), (255, 0, 255), 3)
            cv2.circle(img, (clx, cly), 15, (255, 0, 255), cv2.FILLED)
            
            if calibrated:
                #length = math.hypot(x2 - x1, y2 - y1, z2-z1)
                current_dist =  math.sqrt((gx2-gx1)**2 + (gy2-gy1)**2 + (gz2-gz1)**2) #Scale
                current_vec = np.array([gx2-gx1, gy2-gy1, gz2-gz1]) #Rotation
                current_pos = np.array([cgx, cgy, cgz])  #Translation
                #norm_vec = current_vec/np.linalg.norm(current_vec)

        #FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        #Draw 3D plot
        cv2.imshow("Video",img)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('c'):
            calibrating = True
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
