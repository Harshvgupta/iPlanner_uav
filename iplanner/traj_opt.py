import torch
from snap1 import UAVTrajectoryPlanner
from snap1 import TrajectoryUtils
from closedform import MinimumSnapTrajectoryPlanner
import time
import torch
from torch.cuda import Event

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

torch.set_default_dtype(torch.float32)


class TimingContext:
    def __init__(self, name, use_cuda=False):
        self.name = name
        self.use_cuda = use_cuda
        self.start = None
        self.end = None

    def __enter__(self):
        if self.use_cuda:
            self.start = Event(enable_timing=True)
            self.end = Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda:
            self.end.record()
            torch.cuda.synchronize()
            elapsed_time = self.start.elapsed_time(self.end) / 1000  # Convert to seconds
        else:
            self.end = time.perf_counter()
            elapsed_time = self.end - self.start
        print(f"{self.name} took {elapsed_time:.6f} seconds")


"""closed from solution"""

class TrajOpt:
    def __init__(self):
        self.total_time = None
        self.poly_order = 5
        self.start_vel = torch.zeros(3, dtype=torch.float32)
        self.start_acc = torch.zeros(3, dtype=torch.float32)
        self.end_vel = torch.zeros(3, dtype=torch.float32)
        self.end_acc = torch.zeros(3, dtype=torch.float32)

    def interpolate_trajectory(self, coeffs_x, coeffs_y, coeffs_z, time_stamps, num_points):
        """
        Interpolates polynomial trajectories for x, y, and z coordinates.

        Args:
            coeffs_x (torch.Tensor): Tensor of x polynomial coefficients
            coeffs_y (torch.Tensor): Tensor of y polynomial coefficients
            coeffs_z (torch.Tensor): Tensor of z polynomial coefficients
            time_stamps (torch.Tensor): Tensor of time stamps
            num_points (int): Total number of points to interpolate along the entire trajectory

        Returns:
            torch.Tensor: Interpolated points
        """
        device = coeffs_x.device
        batch_size = coeffs_x.shape[0]

        # Create batch-specific time ranges
        start_times = time_stamps[:, 0]
        end_times = time_stamps[:, -1]
        times = torch.zeros(batch_size, num_points, device=device)

        for i in range(batch_size):
            times[i] = torch.linspace(start_times[i].item(), end_times[i].item(), num_points, device=device)

        x_values = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_x, time_stamps, times, 0)
        y_values = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_y, time_stamps, times, 0)
        z_values = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_z, time_stamps, times, 0)

        x_vel = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_x, time_stamps, times, 1)
        y_vel = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_y, time_stamps, times, 1)
        z_vel = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_z, time_stamps, times, 1)

        x_acc = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_x, time_stamps, times, 2)
        y_acc = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_y, time_stamps, times, 2)
        z_acc = self.min_snap_planner.evaluate_polynomials_vectorized(coeffs_z, time_stamps, times, 2)

        return torch.stack([x_values, y_values, z_values], dim=-1),torch.stack([x_vel, y_vel, z_vel], dim=-1),torch.stack([x_acc, y_acc, z_acc], dim=-1), times

    def odom_callback(self, odom):   
        self.odom = odom

    def target_callback(self, waypoint): 
        self.target = waypoint

    def TrajGeneratorFromPFreeRot(self, preds, step):
        """
        Generates trajectory from predictions with free rotation.

        Args:
            preds (torch.Tensor): Predictions tensor
            step (int): Step number
            v (float): Average drone speed
            target (PoseStamped): Target position
            odom (Odometry): Current drone's position

        Returns:
            torch.Tensor: Interpolated points
            list: timestamps of each waypoint
        """
        use_cuda = preds.is_cuda
        batch_size, num_p, dims = preds.shape
        preds=preds.requires_grad_()
        points_preds = torch.cat((torch.zeros(batch_size, 1, dims, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1)
        # points_preds.register_hook(lambda grad: print("points_preds grad:", grad))
        num_p = num_p + 1

        # Assign total time according to the distance from the drone to the target
        # Assume average drone speed is v = 1 m/s
        # initial guess of the total time t = s/v
        s = torch.norm(points_preds[0, -1, :])  # In robot frame, the last point is the target, and robot position is 0
        v = 1.5
        total_time = s / v
        self.total_time = total_time
        self.min_snap_planner = MinimumSnapTrajectoryPlanner(self.total_time, self.poly_order, self.start_vel, self.start_acc, self.end_vel, self.end_acc)

        all_polys_x, all_polys_y, all_polys_z, all_optimized_times = [], [], [], []
        for i in range(batch_size):
            waypoints = points_preds[i].permute(1, 0)
            polys_x, polys_y, polys_z, optimized_times = self.min_snap_planner.forward(waypoints, total_time)
            all_polys_x.append(polys_x)
            all_polys_y.append(polys_y)
            all_polys_z.append(polys_z)
            all_optimized_times.append(optimized_times)

        all_polys_x = torch.stack(all_polys_x)
        all_polys_y = torch.stack(all_polys_y)
        all_polys_z = torch.stack(all_polys_z)
        all_optimized_times = torch.stack(all_optimized_times)

        # According to control frequency (dt) and distance, calculate num_points
        dt = 0.02
        num_points = round((s / (v * dt) + 1).item())

        interpolated_points,vel,acc,times = self.interpolate_trajectory(all_polys_x, all_polys_y, all_polys_z, all_optimized_times, num_points)

        if torch.isnan(interpolated_points).any():
            print("NaNs detected in TrajGeneratorFromPFreeRot interpolated_trajs")


        # Monitoring message
        rospy.logdebug(f"Distance to target: {s}")
        rospy.logdebug(f"points_preds: {points_preds}")
        rospy.logdebug(f"times: {times}")
        rospy.logdebug(times.shape)


        return interpolated_points,vel,acc,times








