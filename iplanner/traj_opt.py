import torch
from snap1 import UAVTrajectoryPlanner
from snap1 import TrajectoryUtils
import time
import torch
from torch.cuda import Event
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
class TrajOpt:
    def __init__(self):
        super(TrajOpt, self).__init__()
        total_time = 50.0
        poly_order = 5
        start_vel = torch.zeros(3, dtype=torch.float64)
        start_acc = torch.zeros(3, dtype=torch.float64)
        end_vel = torch.zeros(3, dtype=torch.float64)
        end_acc = torch.zeros(3, dtype=torch.float64)
        self.min_snap_planner = UAVTrajectoryPlanner(total_time, poly_order, start_vel, start_acc, end_vel, end_acc)

    

    def interpolate_trajectory(self,coeffs_x, coeffs_y, coeffs_z, time_stamps, num_points):
        """
        Interpolate polynomial trajectories for x, y, and z coordinates.
        
        Args:
        coeffs_x (torch.Tensor): Tensor of shape [batch_size, num_coeffs, num_segments] containing x polynomial coefficients
        coeffs_y (torch.Tensor): Tensor of shape [batch_size, num_coeffs, num_segments] containing y polynomial coefficients
        coeffs_z (torch.Tensor): Tensor of shape [batch_size, num_coeffs, num_segments] containing z polynomial coefficients
        time_stamps (torch.Tensor): Tensor of shape [batch_size, num_waypoints] containing time stamps
        num_points (int): Total number of points to interpolate along the entire trajectory for each batch
        
        Returns:
        torch.Tensor: Interpolated points of shape [batch_size, num_points, 3]
        """
        device = coeffs_x.device
        batch_size = coeffs_x.shape[0]
        # coeffs_x.requires_grad_(True)
        # coeffs_y.requires_grad_(True)
        # coeffs_z.requires_grad_(True)
        # time_stamps.requires_grad_(True)
        
        # Create batch-specific time ranges
        start_times = time_stamps[:, 0]
        end_times = time_stamps[:, -1]
        times = torch.stack([torch.linspace(start, end, num_points, device=device) for start, end in zip(start_times, end_times)])
        
        x_values = TrajectoryUtils.evaluate_polynomials_vectorized(coeffs_x, time_stamps, times, 0)
        y_values = TrajectoryUtils.evaluate_polynomials_vectorized(coeffs_y, time_stamps, times, 0)
        z_values = TrajectoryUtils.evaluate_polynomials_vectorized(coeffs_z, time_stamps, times, 0)
        
        return torch.stack([x_values, y_values, z_values], dim=-1)
    def TrajGeneratorFromPFreeRot(self, preds, step):
        use_cuda = preds.is_cuda
        # with TimingContext("Total TrajGeneratorFromPFreeRot", use_cuda):
        batch_size, num_p, dims = preds.shape
        preds = preds.requires_grad_()
        # with TimingContext("Preprocessing", use_cuda):
        points_preds = torch.cat((torch.zeros(batch_size, 1, dims, device=preds.device,  requires_grad=preds.requires_grad), preds), axis=1)
        num_p = num_p + 1
        total_time = 50.0

    # with TimingContext("Min snap planning", use_cuda):
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
        

    # with TimingContext("Trajectory interpolation", use_cuda):
        num_points = 51
        interpolated_points = self.interpolate_trajectory(all_polys_x, all_polys_y, all_polys_z, all_optimized_times, num_points)
        
        interpolated_points[:, 0, :] = points_preds[:, 0, :]  # Start point
        interpolated_points[:, -1, :] = points_preds[:, -1, :]  # End point

        if torch.isnan(interpolated_points).any():
            print("NaNs detected in TrajGeneratorFromPFreeRot interpolated_trajs")

        return interpolated_points
    
    











