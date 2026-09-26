#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and
`/drive` as the output topic and everything else is yours to change - rewrite
this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then asks for a slow constant speed with the wheels straight. It
will set off from the grid and into the first thing in front of it. That is
deliberate and it is the whole point: **there is no algorithm here and no
algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the workspace, the topics and
your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

What you are allowed to read (see docs/06-rules.md):
    /scan               LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom   ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map            the static map
What you publish:
    /drive              AckermannDriveStamped - you ask for a SPEED, and the
                        simulator closes that loop for you
    /driver/...         anything of your own, for visualisation
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p crawl_speed:=2.0
        # Add your own as you go; config/driver_params.yaml loads them.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('crawl_speed', 1.0)        # [m/s]
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here

        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]
        self.prev_steering = 0.0
        self.prev_speed_cmd = 0.0

        # Recovery states
        # Used when the car gets too close to a wall and needs
        # to reverse out.
        self.recovery_count = 0
        self.recovery_direction = 1.0

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        self._marker_divisor = 0
        self.get_logger().warn(
            'team_driver is up, but this template has no driving logic: it will '
            'crawl straight ahead until it hits something. Implement plan().')

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (about 40 Hz).
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """
        LiDAR reactive controller.

        Main ideas:
        1. Measure free space in several directions.
        2. Try to stay roughly centered between the two sides.
        3. Anticipate turns using diagonal LiDAR regions.
        4. Slow down when turning or approaching obstacles.
        5. Reverse automatically if the car gets stuck.
        6. Smooth steering and speed commands.
        """

        # ============================================================
        # 1. Filter valid LiDAR measurements
        # ============================================================

        valid = (
            np.isfinite(ranges) &
            (ranges > 0.05)
        )

        if not np.any(valid):
            # If LiDAR completely fails, stop.
            return 0.0, 0.0


        # ============================================================
        # 2. Helper function:
        #    get representative distance in an angular region
        # ============================================================

        def sector_distance(angle_min_deg, angle_max_deg, percentile=30):

            mask = (
                (angles >= np.deg2rad(angle_min_deg)) &
                (angles <= np.deg2rad(angle_max_deg)) &
                valid
            )

            if not np.any(mask):
                return self.max_range

            values = ranges[mask]

            return float(
                np.percentile(values, percentile)
            )


        # ============================================================
        # 3. Divide LiDAR into regions
        #
        #
        #              far left
        #                  /
        #             left/
        #                /
        #        ---------------->
        #                \
        #            right\
        #                  \
        #              far right
        #
        # ============================================================

        front_narrow = sector_distance(-8, 8, 20)
        front = sector_distance(-18, 18, 25)

        left_front = sector_distance(18, 45, 30)
        right_front = sector_distance(-45, -18, 30)

        left_side = sector_distance(55, 85, 35)
        right_side = sector_distance(-85, -55, 35)

        far_left = sector_distance(30, 70, 35)
        far_right = sector_distance(-70, -30, 35)


        # ============================================================
        # 4. Recovery mode
        #
        # If we previously detected that the car was almost touching
        # a wall, reverse for several frames.
        # ============================================================

        if self.recovery_count > 0:

            self.recovery_count -= 1

            # Reverse slowly.
            speed_cmd = -0.7

            # Reverse steering direction so that the nose moves
            # away from the obstacle.
            steering_cmd = (
                -0.38 * self.recovery_direction
            )

            self.prev_steering = steering_cmd
            self.prev_speed_cmd = speed_cmd

            return (
                float(steering_cmd),
                float(speed_cmd)
            )


        # ============================================================
        # 5. Detect near-collision
        # ============================================================

        if front_narrow < 0.45:

            # Estimate which side offers more escape space.
            left_escape = (
                0.6 * left_front +
                0.4 * far_left
            )

            right_escape = (
                0.6 * right_front +
                0.4 * far_right
            )

            if left_escape > right_escape:
                self.recovery_direction = 1.0
            else:
                self.recovery_direction = -1.0

            # Roughly half a second depending on scan frequency.
            self.recovery_count = 20

            speed_cmd = -0.7

            steering_cmd = (
                -0.38 * self.recovery_direction
            )

            self.prev_steering = steering_cmd
            self.prev_speed_cmd = speed_cmd

            return (
                float(steering_cmd),
                float(speed_cmd)
            )


        # ============================================================
        # 6. Centering controller
        #
        # If:
        #
        # left distance  = 3 m
        # right distance = 1 m
        #
        # then the car is too close to the RIGHT wall.
        #
        # left - right > 0
        #
        # therefore steer LEFT.
        # ============================================================

        side_error = left_side - right_side

        # Avoid huge corrections caused by strange LiDAR geometry.
        side_error = np.clip(
            side_error,
            -2.0,
            2.0
        )

        centering_gain = 0.10

        centering_steering = (
            centering_gain * side_error
        )


        # ============================================================
        # 7. Turn anticipation
        #
        # Compare diagonal free space.
        #
        # If left-front becomes much more open than right-front,
        # the track probably curves left.
        # ============================================================

        turn_error = far_left - far_right

        turn_error = np.clip(
            turn_error,
            -3.0,
            3.0
        )

        turn_gain = 0.055

        turn_steering = (
            turn_gain * turn_error
        )


        # ============================================================
        # 8. Normal steering
        # ============================================================

        steering_cmd = (
            centering_steering +
            turn_steering
        )


        # ============================================================
        # 9. Obstacle avoidance
        #
        # As the front becomes blocked, obstacle avoidance gradually
        # becomes more important than simple centering.
        # ============================================================

        if front < 2.2:

            left_space = (
                0.7 * left_front +
                0.3 * far_left
            )

            right_space = (
                0.7 * right_front +
                0.3 * far_right
            )

            direction_error = (
                left_space - right_space
            )

            direction_error = np.clip(
                direction_error,
                -3.0,
                3.0
            )

            obstacle_gain = 0.10

            avoidance_steering = (
                obstacle_gain * direction_error
            )

            # Blend normal steering with obstacle steering.
            danger = np.clip(
                (2.2 - front) / 1.5,
                0.0,
                1.0
            )

            steering_cmd = (
                (1.0 - danger) * steering_cmd +
                danger * avoidance_steering
            )


        # ============================================================
        # 10. Strong turn if obstacle is extremely close
        # ============================================================

        if front < 0.9:

            if left_front > right_front:
                steering_cmd = 0.38
            else:
                steering_cmd = -0.38


        # ============================================================
        # 11. Steering limit
        # ============================================================

        max_steering = 0.42

        steering_cmd = np.clip(
            steering_cmd,
            -max_steering,
            max_steering
        )


        # ============================================================
        # 12. Steering smoothing
        #
        # Without smoothing:
        #
        # +0.3
        # -0.3
        # +0.3
        # -0.3
        #
        # can cause oscillation.
        #
        # Low-pass filtering:
        #
        # new = alpha * desired
        #       + (1-alpha) * previous
        # ============================================================

        steering_alpha = 0.30

        steering_cmd = (
            steering_alpha * steering_cmd +
            (1.0 - steering_alpha) *
            self.prev_steering
        )


        # ============================================================
        # 13. Speed control
        # ============================================================

        steering_ratio = (
            abs(steering_cmd) /
            max_steering
        )

        # Straight -> faster
        # Sharp turn -> slower

        max_speed = 3.0
        min_speed = 0.8

        speed_cmd = (
            max_speed -
            (max_speed - min_speed) *
            steering_ratio
        )


        # ============================================================
        # 14. Reduce speed when obstacle approaches
        # ============================================================

        if front < 3.0:
            speed_cmd = min(
                speed_cmd,
                2.2
            )

        if front < 2.0:
            speed_cmd = min(
                speed_cmd,
                1.5
            )

        if front < 1.2:
            speed_cmd = min(
                speed_cmd,
                0.9
            )


        # ============================================================
        # 15. Speed smoothing
        # ============================================================

        speed_alpha = 0.35

        speed_cmd = (
            speed_alpha * speed_cmd +
            (1.0 - speed_alpha) *
            self.prev_speed_cmd
        )


        # ============================================================
        # 16. Store controller state
        # ============================================================

        self.prev_steering = steering_cmd
        self.prev_speed_cmd = speed_cmd


        # ============================================================
        # 17. Return ROS command
        # ============================================================

        return (
            float(steering_cmd),
            float(speed_cmd)
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
