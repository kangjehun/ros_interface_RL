#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16, Bool

import argparse
import pathlib

from dreamer.racing_agent import RacingDreamer

from scipy.signal import medfilt

class AgentNode:

    def __init__(self, algorithm, checkpoint, actor_version):
        # Variables
        self._algorithm = algorithm
        self._checkpoint = checkpoint
        self._actor_version = actor_version
        self._agent = None
        self._laserscan_last_time = rospy.Time(0)
        self._first_scan = True
        self._steer = 0.0 
        self._motor = 0.0
        self._steer_cmd = 1500 
        self._motor_cmd = 1500
        self._observation = dict()
        self._state = None
        # Parameters
        self.init_params()
        # Publishers
        self._steer_pub = rospy.Publisher('/auto_cmd/steer', Int16, queue_size=1)
        self._throttle_pub = rospy.Publisher('/auto_cmd/throttle', Int16, queue_size=1)
        self._auto_mode_pub = rospy.Publisher('/auto_mode', Bool, queue_size=1)
        self._scan_processed_pub = rospy.Publisher('/scan_processed', LaserScan, queue_size=1)
        # Subscribers
        self._scan_sub = rospy.Subscriber(self._scan_topic, LaserScan, self.laser_callback)
        # Initial Setup
        self.initial_setup()
        rospy.loginfo("Initializing Agent Node...")
        rospy.loginfo("- Algorithm : {}".format(self._algorithm))
        rospy.loginfo("- Checkpoint path : {}".format(self._checkpoint))
        rospy.loginfo("- Actor Version : {}".format(self._actor_version))
        try:
            self._agent = RacingDreamer(checkpoint_dir=pathlib.Path(self._checkpoint), actor_version=self._actor_version)
            rospy.loginfo("RacingDreamer initialized successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize RacingDreamer: {e}")
            self._agent = None
        rospy.on_shutdown(self.on_shutdown)

    def on_shutdown(self):
        rospy.loginfo("Shutting down Agent Node, setting auto_mode to False")
        self._auto_mode_pub.publish(Bool(data=False))

    def init_params(self):
        # Parameters
        self._steer_input_min = rospy.get_param('steer_input_min', -1.0)
        self._steer_input_max = rospy.get_param('steer_input_max', 1.0)
        self._steer_output_min = rospy.get_param('steer_output_min', 1100)
        self._steer_output_max = rospy.get_param('steer_output_max', 1900)
        self._motor_input_min = rospy.get_param('motor_input_min', -1.0)
        self._motor_input_max = rospy.get_param('motor_input_max', 1.0)
        self._motor_output_min = rospy.get_param('motor_output_min', 1100)
        self._motor_output_max = rospy.get_param('motor_output_max', 1900)
        self._scan_topic = rospy.get_param('scan_topic', "/scan")
        self._verbose = rospy.get_param('verbose', True)
        # Tuning parameters
        self._config_acc = rospy.get_param('config_acc', 13)
        self._config_dec = rospy.get_param('config_dec', 10)
        self._median_filter = rospy.get_param('median_filter', True)
        self._lpf_steer = rospy.get_param('lpf_steer', True)
        self._lpf_motor = rospy.get_param('lpf_motor', True)

    def initial_setup(self):
        # Set auto mode
        if self._verbose:
            rospy.loginfo("Setting auto mode to True")
        self._auto_mode_pub.publish(Bool(data=False))
        # Set neutral steer and throttle
        if self._verbose:
            rospy.loginfo("Setting initial neutral steer and throttle")
        self._steer_cmd, self._motor_cmd = self.map_actions_to_commands(0.0, 0.0)
        self._steer_pub.publish(Int16(data=self._steer_cmd))
        self._throttle_pub.publish(Int16(data=self._motor_cmd))

    def laser_callback(self, scan_msg):
        # Handle data frequency
        since_last_laserscan = scan_msg.header.stamp - self._laserscan_last_time
        if since_last_laserscan.to_sec() < 0.1 : # limit to 10Hz
            return
        self._laserscan_last_time = scan_msg.header.stamp
        if self._verbose:
            rospy.loginfo('Lidar scan rate: {:.2f} Hz'.format(1.0 / since_last_laserscan.to_sec()))
        # Check Lidar scan raw data
        if self._first_scan :
            assert len(scan_msg.ranges) == 1081, \
                "Error: Expected 1081 LIDAR ranges, but got {}".format(len(scan_msg.ranges))
            self._first_scan = False
        # Process the original lidar scan data
        processed_ranges = self.process_lidar_data(scan_msg.ranges)
        # Create processed lidar scan data for debugging
        processed_scan_msg = LaserScan()
        processed_scan_msg.header = scan_msg.header
        processed_scan_msg.angle_min = scan_msg.angle_min
        processed_scan_msg.angle_max = scan_msg.angle_max
        processed_scan_msg.angle_increment = scan_msg.angle_increment
        processed_scan_msg.time_increment = scan_msg.time_increment
        processed_scan_msg.scan_time = scan_msg.scan_time
        processed_scan_msg.range_min = scan_msg.range_min
        processed_scan_msg.range_max = scan_msg.range_max
        processed_scan_msg.ranges = np.flip(processed_ranges)
        processed_scan_msg.intensities = scan_msg.intensities
        # Derive action from Actor TODO
        steering_pred = 0.0
        motor_pred = 0.0
        if self._agent :
            before = rospy.Time.now()
            motor_pred, steering_pred = self.get_action(processed_ranges[0:1080, ], self._state) # 1081 -> 1080
            after = rospy.Time.now()
            duration = after - before
            if self._verbose:
                rospy.loginfo("pred: steer [-1, 1]: {}, motor [0.005, 1]: {}".format(steering_pred, motor_pred))
                rospy.loginfo('policy takes: {}s'.format(duration.to_sec()))
        else :
            rospy.logwarn("inference model is not initialized yet")
            self.publish_commands(1500, 1500)
            return
        # Apply LPF for steering
        div = 20
        val = 20
        if self._lpf_steer:
            proc_ranges = np.clip(processed_ranges, None, 4) # clip LIDAR distances at max, 4m
            if not self._median_filter:
                proc_ranges[3:-3] = 1.0/3*(proc_ranges[3:-3] + proc_ranges[2:-4] + proc_ranges[4:-2])
            forward_max = max(proc_ranges[int(1080/2 - 150):int(1080/2+150)])
            val = 18 - forward_max * 3
        self._steer = float(self._steer) * (div - val) / (div) + float(steering_pred) * (val) / (div)
        # Apply LPF for motor (pseudo-lpf)
        if self._lpf_motor:
            if float(motor_pred) < 0.5 :
                self._motor = self._motor - float(self._config_dec)/1000
            else :
                self._motor = self._motor + float(self._config_acc)/1000
            # clip with the reduced action space of dreamer agent
            if self._motor > 0.120 :
                self._motor = 0.120
            if self._motor < 0.080 :
                self._motor = 0.080
        else :
            self._motor = motor_pred
        rospy.loginfo("proc: steer [-1, 1]: {}, motor [0.005, 1]: {}".format(self._steer, self._motor)) 
        # Map actions to commands
        self._steer_cmd, self._throttle_cmd = self.map_actions_to_commands(self._steer, self._motor)
        # Publish commands
        self.publish_commands(self._steer_cmd, self._throttle_cmd)
        self._scan_processed_pub.publish(processed_scan_msg)
        self._auto_mode_pub.publish(Bool(data=True))
    
    def process_lidar_data(self, ranges, kernel_size=5):
        # Change increment direction (CCW -> CW)
        proc_ranges = np.flip(np.array(ranges))
        # Interpolate nan values and apply a median filter
        if self._median_filter :
            nans = np.isnan(proc_ranges)
            nan_idx = np.where(nans)
            idx_nonan = np.where(~nans)[0]
            value_nonan = proc_ranges[~nans]
            proc_ranges = np.interp(nan_idx, idx_nonan, value_nonan)
            proc_ranges = medfilt(proc_ranges, kernel_size=5)
        return proc_ranges
    
    def get_action(self, processed_ranges, state):
        self._observation['lidar'] = processed_ranges
        assert processed_ranges.shape == (1080,), \
        f"Error: Expected processed_ranges to have shape (1080,), but got {processed_ranges.shape}"
        action, self._state = self._agent.action(self._observation, self._state)
        pred_motor = action['motor']
        pred_steer = action['steer']
        # Just for debugging
        # pred_motor = 0
        # pred_steer = 0
        return pred_motor, pred_steer
    
    def map_actions_to_commands(self, steer, motor):
        si_min = self._steer_input_min
        si_max = self._steer_input_max
        so_min = self._steer_output_min
        so_max = self._steer_output_max
        mi_min = self._motor_input_min
        mi_max = self._motor_input_max
        mo_min = self._motor_output_min
        mo_max = self._motor_output_max
        # Map steering [steer_input_min, steer_input_max] to [steer_output_min, steer_output_max]
        steer_cmd = int(((steer - si_min) / (si_max - si_min)) * (so_max - so_min) + so_min)
        # Map motor [motor_input_min, motor_input_max] to [motor_output_max, motor_output_min]
        throttle_cmd = int(((mi_max - motor) / (mi_max - mi_min)) * (mo_max - mo_min) + mo_min)
        # Clipping the commands to the given ROS parameters
        steer_cmd = np.clip(steer_cmd, so_min, so_max)
        throttle_cmd = np.clip(throttle_cmd, mo_min, mo_max)
        return steer_cmd, throttle_cmd

    def publish_commands(self, steer_cmd, throttle_cmd):
        if self._verbose:
            rospy.loginfo("steer_cmd [1100, 1900]: {}, motor [1100, 1900]: {}".format(steer_cmd, throttle_cmd)) 
        self._steer_pub.publish(Int16(data=steer_cmd))
        self._throttle_pub.publish(Int16(data=throttle_cmd))
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    algorithm = rospy.get_param('algorithm', 'dreamer')
    checkpoint = rospy.get_param('checkpoint', '../checkpoint/austria_dreamer')
    actor_version = rospy.get_param('actor_version', 'normalized')
    # Execute ROS node
    rospy.init_node('agent_node', anonymous=True)
    agent_node = AgentNode(algorithm, checkpoint, actor_version)
    agent_node.run()