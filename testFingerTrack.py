import cv2
import mediapipe as mp
import time

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
                cx,cy = int(lm.x*w), int(lm.y*h) #center
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])
        return lmlist
    
def calibrate(lmlist):
    """
    Calibration using samples
    """
    print("Calibrating")
    samples = 100
    x1_sum, y1_sum, z1_sum, x2_sum, y2_sum,z2_sum = 0
    x1_avg, y1_avg, z1_avg, x2_avg, y2_avg, z2_avg = 0
    
    for i in range(samples):
        x1_sum += lmlist[4][1]
        y1_sum += lmlist[4][2]
        z1_sum += lmlist[4][3]
        x2_sum += lmlist[8][1]
        y2_sum += lmlist[8][2]
        z2_sum += lmlist[8][3]

        
    x1_avg = x1_sum/samples
    x2_avg = x2_sum/samples
    y1_avg = y1_sum/samples
    y2_avg = y2_sum/samples
    z1_avg = z1_sum/samples
    z2_avg = z2_sum/samples
    
    cx_avg, cy_avg, cz_avg = (x1_avg + x2_avg) // 2, (y1_avg + y2_avg) // 2, (z1_avg + z2_avg) // 2
    pos_avg = [cx_avg, cy_avg, cz_avg] #Pos
    dist_avg = math.sqrt((x2_avg-x1_avg)**2 + (y2_avg-y1_avg)**2 + (z2_avg-z1_avg)**2) #Scale
    vec_avg = [x2_avg - x1_avg, y2_avg - y1_avg, z2_avg - z1_avg] #Rotation
    
    print("Calibrated")
    return pos_avg, dist_avg, vec_avg

def main():
    #Frame rates
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    cal_x1, cal_x2, cal_y1, cal_y2, cal_z1, cal_z2 = calibrate()
    
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            x1, y1, z1 = lmlist[4][1], lmlist[4][2], lmlist[4][3]
            x2, y2, z1 = lmlist[8][1], lmlist[8][2], lmlist[4][3]
            cx, cy, cz = (x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2
            
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            #length = math.hypot(x2 - x1, y2 - y1, z2-z1)
            current_dist =  math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) #Scale
            current_vec = [x2-x1, y2-y1, z2-z1]#Rotation
            current_pos = [cx, cy,cz]  #Translation
            print(current_dist)
            
        #FPS
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Video",img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
