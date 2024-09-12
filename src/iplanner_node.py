# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
import time
from std_msgs.msg import Float32, Int16
import numpy as np
from sensor_msgs.msg import Image, Joy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist, Pose
from std_msgs.msg import Float32MultiArray
import ros_numpy

rospack = rospkg.RosPack()
pack_path = rospack.get_path('iplanner_node')
planner_path = os.path.join(pack_path,'iplanner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from iplanner.ip_algo import IPlannerAlgo
from iplanner.rosutil import ROSArgparse

class iPlannerNode:
    def __init__(self, args):
        super(iPlannerNode, self).__init__()
        self.config(args)

        # init planner algo class
        self.iplanner_algo = IPlannerAlgo(args=args)
        self.tf_listener = tf.TransformListener()
        
        rospy.sleep(2.5) # wait for tf listener to be ready

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0
        self.is_goal_processed = False
        self.is_smartjoy = False

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # process time
        self.timer_data = Float32()
        
        rospy.Subscriber(self.image_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        timer_topic = '/ip_timer'
        status_topic = '/ip_planner_status'
        
        # planning status topics
        self.timer_pub = rospy.Publisher(timer_topic, Float32, queue_size=10)
        self.status_pub = rospy.Publisher(status_topic, Int16, queue_size=10)

        # self.path_pub  = rospy.Publisher(self.path_topic, Path, queue_size=10)
        # self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", Path, queue_size=10)
        self.path_pub = rospy.Publisher(self.path_topic, MultiDOFJointTrajectory, queue_size=1)
        self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", MultiDOFJointTrajectory, queue_size=1)
        self.path_vis_pub = rospy.Publisher(self.path_vis_topic, Path, queue_size=1)
        
        # timestamp for each waypoint
        self.wp_time = None

        rospy.loginfo("iPlanner Ready.")
        

    def config(self, args):
        self.main_freq   = args.main_freq
        self.model_save  = args.model_save
        self.image_topic = args.depth_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic
        self.path_vis_topic = args.path_topic + "_vis"
        self.path_time_topic = args.path_topic + "_time"
        self.frame_id    = args.robot_id
        self.world_id    = args.world_id
        self.uint_type   = args.uint_type
        self.image_flip  = args.image_flip
        self.conv_dist   = args.conv_dist
        self.depth_max   = args.depth_max
        # fear reaction
        self.is_fear_act = args.is_fear_act
        self.buffer_size = args.buffer_size
        self.ang_thred   = args.angular_thred
        self.track_dist  = args.track_dist
        self.joyGoal_scale = args.joyGoal_scale
        return 

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning and self.is_goal_init:
                # main planning starts
                cur_image = self.img.copy()
                start = time.time()
                # Network Planning
                # self.preds, self.waypoints, fear_output, _ = self.iplanner_algo.plan(cur_image, self.goal_rb)
                # end = time.time()
                # self.timer_data.data = (end - start) * 1000
                # self.timer_pub.publish(self.timer_data)

                self.preds, self.waypoints, self.velocities, self.accelerations, self.wp_time, fear_output, _ = self.iplanner_algo.plan(cur_image, self.goal_rb)
                end = time.time()
                self.timer_data.data = (end - start) * 1000
                self.timer_pub.publish(self.timer_data)
                # check goal less than converage range
                if (np.sqrt(self.goal_rb[0][0]**2 + self.goal_rb[0][1]**2) < self.conv_dist) and self.is_goal_processed and (not self.is_smartjoy):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    # planner status -> Success
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)

                    rospy.loginfo("Goal Arrived")
                self.fear = torch.tensor([[0.0]], device=fear_output.device)
                if self.is_fear_act:
                    self.fear = fear_output
                    is_track_ahead = self.isForwardTraking(self.waypoints)
                    self.fearPathDetection(self.fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "current path prediction is invaild.")
                        # planner status -> Fails
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)
                # self.pubPath(self.waypoints, self.is_goal_init)
                self.pubPath(self.waypoints, self.velocities, self.accelerations, self.is_goal_init)
            r.sleep()
        rospy.spin()

    # def pubPath(self, waypoints, is_goal_init=True):
    #     path = Path()
    #     fear_path = Path()
    #     if is_goal_init:
    #         for p in waypoints.squeeze(0):
    #             pose = PoseStamped()
    #             pose.pose.position.x = p[0]
    #             pose.pose.position.y = p[1]
    #             pose.pose.position.z = p[2]
    #             path.poses.append(pose)
    #     # add header
    #     path.header.frame_id = fear_path.header.frame_id = self.frame_id
    #     path.header.stamp = fear_path.header.stamp = self.image_time
    #     # publish fear path
    #     if self.is_fear_reaction:
    #         fear_path.poses = path.poses.copy()
    #         path.poses = path.poses[:1]
    #     # publish path
    #     self.fear_path_pub.publish(fear_path)
    #     self.path_pub.publish(path)
    #     return
    # def pubPath(self, waypoints, velocities, accelerations, is_goal_init=True):
    #     path = Path()
    #     fear_path = Path()
    #     if is_goal_init:
    #         for p, v, a in zip(waypoints.squeeze(0), velocities.squeeze(0), accelerations.squeeze(0)):
    #             pose_vel_acc = PoseVelAcc()
    #             pose_vel_acc.pose.position.x = p[0]
    #             pose_vel_acc.pose.position.y = p[1]
    #             pose_vel_acc.pose.position.z = p[2]
    #             pose_vel_acc.velocity.x = v[0]
    #             pose_vel_acc.velocity.y = v[1]
    #             pose_vel_acc.velocity.z = v[2]
    #             pose_vel_acc.acceleration.x = a[0]
    #             pose_vel_acc.acceleration.y = a[1]
    #             pose_vel_acc.acceleration.z = a[2]
    #             path.poses.append(pose_vel_acc)
    #     # add header
    #     path.header.frame_id = fear_path.header.frame_id = self.frame_id
    #     path.header.stamp = fear_path.header.stamp = self.image_time
    #     # publish fear path
    #     if self.is_fear_reaction:
    #         fear_path.poses = path.poses.copy()
    #         path.poses = path.poses[:1]
    #     # publish path
    #     self.fear_path_pub.publish(fear_path)
    #     self.path_pub.publish(path)
    #     return
    def pubPath(self, waypoints, velocities, accelerations, is_goal_init=True):
        trajectory = MultiDOFJointTrajectory()
        fear_trajectory = MultiDOFJointTrajectory()
        path_vis = Path()  # add path_vis to help visualize the planned path in rviz
        if is_goal_init:
            for p, v, a in zip(waypoints.squeeze(0), velocities.squeeze(0), accelerations.squeeze(0)):
                p = p.detach().cpu().numpy() if p.is_cuda else p.detach().numpy()
                v = v.detach().cpu().numpy() if v.is_cuda else v.detach().numpy()
                a = a.detach().cpu().numpy() if a.is_cuda else a.detach().numpy()
                point = MultiDOFJointTrajectoryPoint()
                
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                transform.translation.z = p[2]

                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path_vis.poses.append(pose)
                
                velocity = Twist()
                velocity.linear.x, velocity.linear.y, velocity.linear.z = v[0], v[1], v[2]
                
                acceleration = Twist()
                acceleration.linear.x, acceleration.linear.y, acceleration.linear.z = a[0], a[1], a[2]
                
                point.transforms.append(transform)
                point.velocities.append(velocity)
                point.accelerations.append(acceleration)
                
                trajectory.points.append(point)
        
        # add header
        trajectory.header.frame_id = fear_trajectory.header.frame_id = self.frame_id
        trajectory.header.stamp = fear_trajectory.header.stamp = self.image_time
        
        path_vis.header.frame_id = self.frame_id
        path_vis.header.stamp = self.image_time
        
        # Transform path_vis to /map
        try:
            for i, pose_stamped in enumerate(path_vis.poses):
                # Transform each pose in path_vis to /map
                if pose_stamped.header.frame_id == "":
                    pose_stamped.header.frame_id = self.frame_id  # Ensure the source frame is set

                # Wait for the transform to be available
                self.tf_listener.waitForTransform("/map", pose_stamped.header.frame_id, rospy.Time(0), rospy.Duration(1.0))

                # Transform pose to /map
                (trans, rot) = self.tf_listener.lookupTransform("/map", self.frame_id, rospy.Time(0))
                transform_matrix = self.tf_listener.fromTranslationRotation(trans, rot)
                position_robot_frame = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z, 1.0])
                transformed_pose = np.dot(transform_matrix, position_robot_frame)
                
                # transformed_pose = self.tf_listener.transformPose("/map", pose_stamped)
                new_pose = PoseStamped()
                new_pose.pose.position.x = transformed_pose[0]
                new_pose.pose.position.y = transformed_pose[1]
                new_pose.pose.position.z = transformed_pose[2]
                path_vis.poses[i] = new_pose

            # Update path_vis header to the new frame
            path_vis.header.frame_id = "/map"
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Transform to /map failed: %s", e)

        # Transform path to /map
        try:
            for i, point in enumerate(trajectory.points): 
                # pose to PoseStamped, vel to TwistStamped, acc to TwistStamped  
                pose = PoseStamped()
                pose.pose.position.x = point.transforms[0].translation.x
                pose.pose.position.y = point.transforms[0].translation.y
                pose.pose.position.z = point.transforms[0].translation.z
                pose.header.frame_id = self.frame_id 

                velocity = TwistStamped()
                velocity.twist.linear.x = point.velocities[0].linear.x
                velocity.twist.linear.y = point.velocities[0].linear.y
                velocity.twist.linear.z = point.velocities[0].linear.z
                velocity.header.frame_id = self.frame_id

                acceleration = TwistStamped()
                acceleration.twist.linear.x = point.accelerations[0].linear.x
                acceleration.twist.linear.y = point.accelerations[0].linear.y
                acceleration.twist.linear.z = point.accelerations[0].linear.z
                acceleration.header.frame_id = self.frame_id    

                # Wait for the transform to be available    
                self.tf_listener.waitForTransform("/map", self.frame_id, rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = self.tf_listener.lookupTransform("/map", self.frame_id, rospy.Time(0))
                transform_matrix = self.tf_listener.fromTranslationRotation(trans, rot)
                # print(f'translation: {transform_matrix[0][3]}, {transform_matrix[1][3]}, {transform_matrix[2][3]}, {transform_matrix[3][3]}')   
                position_robot_frame = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, 1.0])
                transformed_position = np.dot(transform_matrix, position_robot_frame)
                trajectory.points[i].transforms[0].translation.x = transformed_position[0]
                trajectory.points[i].transforms[0].translation.y = transformed_position[1]
                trajectory.points[i].transforms[0].translation.z = transformed_position[2]
                # print(f'transformed position: {transformed_position[0]}, {transformed_position[1]}, {transformed_position[2]}')

                # Transform velocity to /map
                # self.tf_listener.waitForTransform("/map", self.frame_id, rospy.Time(0), rospy.Duration(1.0))
                # (trans, rot) = self.tf_listener.lookupTransform("/map", self.frame_id, rospy.Time(0))
                # transform_matrix = self.tf_listener.fromTranslationRotation(trans, rot)
                linear_velocity = np.array([velocity.twist.linear.x, velocity.twist.linear.y, velocity.twist.linear.z, 0.0])
                transformed_velocity = np.dot(transform_matrix[0:3, 0:3], linear_velocity[0:3])
                trajectory.points[i].velocities[0].linear.x = transformed_velocity[0]
                trajectory.points[i].velocities[0].linear.y = transformed_velocity[1]   
                trajectory.points[i].velocities[0].linear.z = transformed_velocity[2]
                
                # Transform acceleration to /map
                acceleration_vector = np.array([acceleration.twist.linear.x, acceleration.twist.linear.y, acceleration.twist.linear.z, 0.0])
                transformed_acceleration = np.dot(transform_matrix[0:3, 0:3], acceleration_vector[0:3])
                trajectory.points[i].accelerations[0].linear.x = transformed_acceleration[0]
                trajectory.points[i].accelerations[0].linear.y = transformed_acceleration[1]    
                trajectory.points[i].accelerations[0].linear.z = transformed_acceleration[2]    
            # Update path header to the new frame
            trajectory.header.frame_id = "/map"

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Transform to /map failed: %s", e)



        # publish fear trajectory
        if self.is_fear_reaction:
            fear_trajectory.points = trajectory.points.copy()
            trajectory.points = trajectory.points[:1]
        
        # publish trajectory
        self.fear_path_pub.publish(fear_trajectory)
        self.path_pub.publish(trajectory)

        # publish path_vis
        self.path_vis_pub.publish(path_vis)

        # # publish waypoint time

        # # tensor to list
        # waypoint_time = waypoint_time.squeeze(0).detach().cpu().numpy().tolist()
        # waypoint_time_data = Float32MultiArray()
        # waypoint_time_data.data = waypoint_time
        # self.path_time_pub.publish(waypoint_time_data)
        return

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return None

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if phead is None or phead.dot(xhead) > 1.0 - self.ang_thred:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            rospy.loginfo("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            # reset fear reaction
            self.fear_buffter = 0
            self.is_fear_reaction = False
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3]**2 + joy_msg.axes[4]**2) < 1e-3:
                # reset fear reaction
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.ready_for_planning = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.frame_id
                joy_goal.point.x = joy_msg.axes[4] * self.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.joyGoal_scale
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = rospy.Time.now()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False
        return

    def goalCallback(self, msg):
        rospy.loginfo("Recevied a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        # reset fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # reste planner status
        self.planner_status.data = 0
        return

    def imageCallback(self, msg):
        # rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        self.image_time = msg.header.stamp
        frame = ros_numpy.numpify(msg)
        frame[~np.isfinite(frame)] = 0
        if self.uint_type:
            frame = frame / 1000.0
        frame[frame > self.depth_max] = 0.0
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
        # img.show()
        if self.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.img = np.array(frame.transpose(PIL.Image.ROTATE_180))
        else:
            self.img = frame

        if self.is_goal_init:
            goal_robot_frame = self.goal_pose;
            if not self.goal_pose.header.frame_id == self.frame_id:
                try:
                    goal_robot_frame.header.stamp = self.tf_listener.getLatestCommonTime(self.goal_pose.header.frame_id,
                                                                                         self.frame_id)
                    goal_robot_frame = self.tf_listener.transformPoint(self.frame_id, goal_robot_frame)
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("Fail to transfer the goal into base frame.")
                    return
            goal_robot_frame = torch.tensor([goal_robot_frame.point.x, goal_robot_frame.point.y, goal_robot_frame.point.z], dtype=torch.float32)[None, ...]
            self.goal_rb = goal_robot_frame
        else:
            return
        self.ready_for_planning = True
        self.is_goal_processed  = True
        return

if __name__ == '__main__':

    node_name = "iplanner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('main_freq',         type=int,   default=5,                          help="Main frequency of the path planner.")
    parser.add_argument('model_save',        type=str,   default='/models/plannernet.pt',    help="Path to the saved model.")
    parser.add_argument('crop_size',         type=tuple, default=[360,640],                  help='Size to crop the image to.')
    parser.add_argument('uint_type',         type=bool,  default=False,                      help="Determines if the image is in uint type.")
    parser.add_argument('depth_topic',       type=str,   default='/rgbd_camera/depth/image', help='Topic for depth image.')
    parser.add_argument('goal_topic',        type=str,   default='/way_point',               help='Topic for goal waypoints.')
    parser.add_argument('path_topic',        type=str,   default='/path',                    help='Topic for iPlanner path.')
    parser.add_argument('robot_id',          type=str,   default='base',                     help='TF frame ID for the robot.')
    parser.add_argument('world_id',          type=str,   default='odom',                     help='TF frame ID for the world.')
    parser.add_argument('depth_max',         type=float, default=10.0,                       help='Maximum depth distance in the image.')
    parser.add_argument('image_flip',        type=bool,  default=True,                       help='Indicates if the image is flipped.')
    parser.add_argument('conv_dist',         type=float, default=0.5,                        help='Convergence range to the goal.')
    parser.add_argument('is_fear_act',       type=bool,  default=True,                       help='Indicates if fear action is enabled.')
    parser.add_argument('buffer_size',       type=int,   default=10,                         help='Buffer size for fear reaction.')
    parser.add_argument('angular_thred',     type=float, default=0.3,                        help='Angular threshold for turning.')
    parser.add_argument('track_dist',        type=float, default=0.5,                        help='Look-ahead distance for path tracking.')
    parser.add_argument('joyGoal_scale',     type=float, default=0.5,                        help='Scale for joystick goal distance.')
    parser.add_argument('sensor_offset_x',   type=float, default=0.0,                        help='Sensor offset on the X-axis.')
    parser.add_argument('sensor_offset_y',   type=float, default=0.0,                        help='Sensor offset on the Y-axis.')

    args = parser.parse_args()
    args.model_save = planner_path + args.model_save


    node = iPlannerNode(args)

    node.spin()