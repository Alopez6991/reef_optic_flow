#!/usr/bin/env python
# Command line arguments
from optparse import OptionParser
# ROS imports
import roslib, rospy
# math
import math
# opencv imports
import cv2
# numpy imports - basic math and matrix manipulation
import numpy as np
# import from gazebo for modle state
from gazebo_msgs.msg import ModelStates
# imports for ROS image handling
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovariance, TwistWithCovarianceStamped, Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
# message imports specific to this package
from optic_flow_example.msg import OpticFlowMsg
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
################################################################################

def draw_optic_flow_field(gray_image, points, flow,arrow_color1):
    '''
    gray_image: opencv gray image, e.g. shape = (width, height)
    points: points at which optic flow is tracked, e.g. shape = (npoints, 1, 2)
    flow: optic flow field, should be same shape as points
    '''
    color_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    linewidth = 1
    linewidth2 = 2
    for i, point in enumerate(points):
        x = point[0]
        y = point[1]
        vx = flow[i,0]
        vy = flow[i,1]
        # vx2 = flow2[i,0]
        # vy2 = flow2[i,1]

        cv2.arrowedLine(color_img, (x,y), (x+vx, y+vy), arrow_color1, linewidth) # draw a red line from the point with vector = [vx, vy]        
        # cv2.arrowedLine(color_img, (x,y), (x+vx2, y+vy2), arrow_color2, linewidth2) # draw a red line from the point with vector = [vx, vy]        

    cv2.imshow('optic_flow_field',color_img)
    cv2.waitKey(1)

################################################################################
    
def define_points_at_which_to_track_optic_flow(image, spacing):
    points_to_track = []
    scale=0.5
    for x in range(int(image.shape[0]*scale/2),image.shape[0]-int(image.shape[0]*scale/2),spacing):
        for y in range(int(image.shape[1]*scale/2),image.shape[1]-int(image.shape[1]*scale/2),spacing):
            new_point = [y, x]
            points_to_track.append(new_point)
    points_to_track = np.array(points_to_track, dtype=np.float32) # note: float32 required for opencv optic flow calculations
    points_to_track = points_to_track.reshape(points_to_track.shape[0], 1, points_to_track.shape[1]) # for some reason this needs to be shape (npoints, 1, 2)
    return points_to_track

################################################################################

class Optic_Flow_Calculator:
    def __init__(self, topic):
        # Define the source of the images, e.g. rostopic name
        self.image_source = topic

        # Initialize image aquisition
        self.bridge = CvBridge()
        self.prev_image = None
        self.last_time = 0

        # Other initalization terms
        self.frame_idx = 0
        self.tracks = []
        self.detect_interval = 5
        self.count = 0

        # Lucas Kanade Optic Flow parameters
        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        ########################################################################

        # Publishers

        # Lucas Kanade Publisher
        self.optic_flow_pub = rospy.Publisher("optic_flow", OpticFlowMsg, queue_size=10)
        self.optic_flow_pub_meanx = rospy.Publisher("optic_flow_mean_vx", Float32, queue_size=10)
        self.optic_flow_pub_meany = rospy.Publisher("optic_flow_mean_vy", Float32, queue_size=10)
        
        # self.constant=rospy.Publisher("constant")
        ########################################################################

        # Subscribers

        # Raw Image Subscriber
        self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)
        # subscribing to gazebo model, can replace this with mocap later
        self.model_state_sub=rospy.Subscriber('/gazebo/model_states',ModelStates,self.model_state_callback)

        ########################################################################

        # Functions

    def alpha_beta_filter(self,old,new,alpha):
        new = alpha * new + (1-alpha) * old
        old=new
        return new

    def euler_from_quaternion(self,x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    ########################################################################

    # Callback Functions

    def model_state_callback(self,data):
        # pose
        self.px=data.pose[0].position.x
        self.py=data.pose[0].position.y
        self.pz=data.pose[0].position.z
        # print(self.pz)
        # orientation
        self.ox=data.pose[0].orientation.x
        self.oy=data.pose[0].orientation.y
        self.oz=data.pose[0].orientation.z
        self.ow=data.pose[0].orientation.w
        # linear velocity
        self.lx=data.twist[0].linear.x
        self.ly=data.twist[0].linear.y
        self.lz=data.twist[0].linear.z
        # angular velocity
        self.ax=data.twist[0].angular.x
        self.ay=data.twist[0].angular.y
        self.az=data.twist[0].angular.z

    def image_callback(self,image):
        try: # if there is an image
            #self.count+=1
            self.count+=1
            if self.count%2==0:
                # Acquire the image, and convert to single channel gray image
                curr_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="mono8")
                gray = curr_image
                # Get time stamp
                secs = image.header.stamp.secs
                nsecs = image.header.stamp.nsecs
                curr_time = float(secs) + float(nsecs)*1e-9
                
                # If this is the first loop, initialize image matrices
                if self.prev_image is None:
                    self.prev_image = gray
                    self.last_time = curr_time
                    self.points_to_track=cv2.goodFeaturesToTrack(gray,mask = None,**feature_params)
                    return # skip the rest of this loop
                    
                # get time between images
                dt = curr_time - self.last_time
                self.last_time = curr_time
                # print(dt , 'sec')

                # calculate optic flow with lucas kanade
                # see: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html
                new_position_of_tracked_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_image, gray, self.points_to_track, None, **self.lk_params)

                # calculate flow field
                good_new_pts = new_position_of_tracked_points[status==1]
                good_old_pts = self.points_to_track[status==1]
                # self.num_points_pub.publish(np.shape(good_old_pts)[0])
                flow = (good_new_pts-good_old_pts)/dt
                # print(np.shape(good_old_pts)[0])
                self.points_to_track=cv2.goodFeaturesToTrack(gray,mask = None,**feature_params)
                #print(np.size(self.points_to_track))
                self.prev_image = gray
                # draw the flow field
                arrow_color = [0,255,0] # bgr colorspace
                draw_optic_flow_field(curr_image, good_old_pts, flow,arrow_color)
              

                #subscribe to drone Z 
                z=1.0*self.pz #m/s

                #calibration constants
                # fx=907.894773
                # fy=907.894773
                # cx=617.672438
                # cy=356.842024
                fx=381.36246688113556
                fy=381.36246688113556
                cx=320.5
                cy=240.5

                X_dynamics = - self.ay * fx + self.az * good_old_pts[:,1]

                Y_dynamics = self.ax * fy- self.az * good_old_pts[:,0]

                induced_flow = np.matrix([X_dynamics.T,Y_dynamics.T])

                arrow_color = [0,0,255] # bgr colorspace
                arrow_color2 = [0,255,0] # bgr colorspace

                corrected_flow = flow - induced_flow.T 

                mean_sensor_x=np.mean(np.array(corrected_flow[:,0]))
                mean_sensor_y=np.mean(np.array(corrected_flow[:,1]))
                vel_x=Float32()
                vel_y=Float32()
                vel_x.data = mean_sensor_x * self.pz /fx    
                vel_y.data = mean_sensor_y * self.pz /fy

                print(self.pz,fx)

                self.optic_flow_pub_meanx.publish(vel_x)
                self.optic_flow_pub_meany.publish(vel_y)

        except CvBridgeError, e:
                print (e)
            
    def main(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print ("Shutting down")
            cv2.destroyAllWindows()

################################################################################

if __name__ == '__main__':    
    parser = OptionParser()
    parser.add_option("--topic", type="str", dest="topic", default='/multirotor/camera/image_raw',
                        help="ros topic with Float32 message for velocity control")
    (options, args) = parser.parse_args()

    rospy.init_node('optic_flow_calculator', anonymous=True)
    optic_flow_calculator = Optic_Flow_Calculator(options.topic)
    optic_flow_calculator.main()









