import torch
from torch import nn
import qpth
from qpth.qp import QPFunction
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.cuda import Event
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Define the dtype and device at one place
dtype = torch.float64
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'

class TrajectoryUtils:
    @staticmethod
    def log_product_vectorized(seq):
        return torch.exp(torch.sum(torch.log(seq), dim=-1))

    @staticmethod
    def calculate_time_vector_vectorized(time, order, derivative_order):
        batch_size = time.shape[0]
        time_vector = torch.zeros(batch_size, order + 1, dtype=dtype, device=device)
        
        for i in range(derivative_order + 1, order + 2):
            seq = torch.arange(i - derivative_order, i, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1)
            product = TrajectoryUtils.log_product_vectorized(seq)
            time_vector[:, i - 1] = product * (time ** (i - derivative_order - 1))
        
        return time_vector
    @staticmethod
    def ensure_spd_matrix(Q, method='eigenvalue_clip', min_eigenvalue=1e-6):
        """
        Ensure a matrix is symmetric positive definite using various methods.
        
        Args:
        Q (torch.Tensor): Input matrix
        method (str): Method to use ('eigenvalue_clip', 'nearest_spd', or 'cholesky_with_perturbation')
        min_eigenvalue (float): Minimum eigenvalue for 'eigenvalue_clip' method
        
        Returns:
        torch.Tensor: SPD matrix
        """
        if method == 'eigenvalue_clip':
            # Clip eigenvalues to ensure they're all positive
            eigenvalues, eigenvectors = torch.linalg.eigh(Q)
            eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
            return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()

        elif method == 'nearest_spd':
            # Find the nearest SPD matrix
            B = (Q + Q.t()) / 2
            _, s, V = torch.svd(B)
            H = V @ torch.diag(s) @ V.t()
            A2 = (B + H) / 2
            A3 = (A2 + A2.t()) / 2
            return A3

        elif method == 'cholesky_with_perturbation':
            # Attempt Cholesky decomposition, perturb if not SPD
            n = Q.shape[0]
            for k in range(10):  # Max 10 attempts
                try:
                    L = torch.linalg.cholesky(Q)
                    return Q
                except RuntimeError:
                    # If Cholesky fails, add a small perturbation
                    Q = Q + torch.eye(n, device=Q.device, dtype=Q.dtype) * (10.0 ** (-9 + k))
            
            # If still not SPD after 10 attempts, use eigenvalue method as fallback
            return TrajectoryUtils.ensure_spd_matrix(Q, method='eigenvalue_clip', min_eigenvalue=min_eigenvalue)

        else:
            raise ValueError("Unknown method. Use 'eigenvalue_clip', 'nearest_spd', or 'cholesky_with_perturbation'.")

    @staticmethod
    def compute_Q_matrix(poly_order, derivative_order, time_stamps):
        num_segments = len(time_stamps) - 1
        matrix_size = poly_order + 1
        Q_all = torch.zeros((num_segments * matrix_size, num_segments * matrix_size), dtype=dtype, device=device)

        # Compute time differences and powers
        start_times = time_stamps[:-1].unsqueeze(1)
        end_times = time_stamps[1:].unsqueeze(1)
        powers = torch.arange(1, (poly_order - derivative_order) * 2 + 2, dtype=dtype, device=device).unsqueeze(0)
        time_diff_powers = end_times.pow(powers) - start_times.pow(powers)

        # Create indices for vectorized computation
        i_indices = torch.arange(derivative_order + 1, poly_order + 2, dtype=dtype, device=device)
        j_indices = torch.arange(derivative_order + 1, poly_order + 2, dtype=dtype, device=device)
        i_mesh, j_mesh = torch.meshgrid(i_indices, j_indices, indexing='ij')

        k1 = i_mesh - derivative_order - 1
        k2 = j_mesh - derivative_order - 1
        k = k1 + k2 + 1

        # Compute prod_k1 and prod_k2
        prod_k1 = torch.prod(torch.arange(1, derivative_order + 1, dtype=dtype, device=device).unsqueeze(0) + k1.unsqueeze(-1), dim=-1)
        prod_k2 = torch.prod(torch.arange(1, derivative_order + 1, dtype=dtype, device=device).unsqueeze(0) + k2.unsqueeze(-1), dim=-1)

        # Compute Q_matrices for all segments at once
        Q_upper = prod_k1.unsqueeze(0) * prod_k2.unsqueeze(0) / k.unsqueeze(0) * time_diff_powers[:, k.long() - 1]

        # Fill the block diagonal of Q_all
        for i in range(num_segments):
            start_idx = i * matrix_size
            end_idx = (i + 1) * matrix_size
            Q_segment = torch.zeros((matrix_size, matrix_size), dtype=dtype, device=device)
            Q_segment[derivative_order:, derivative_order:] = Q_upper[i]
            Q_all[start_idx:end_idx, start_idx:end_idx] = torch.triu(Q_segment) + torch.triu(Q_segment, 1).transpose(-2, -1)

        return Q_all

    @staticmethod
    def evaluate_polynomial_vectorized(polynomial_coefficients, times, derivative_order):
        polynomial_order = polynomial_coefficients.shape[1] - 1
        powers = torch.arange(polynomial_order + 1, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        time_powers = times.unsqueeze(-1).pow(powers)
        
        if derivative_order <= 0:
            values = (polynomial_coefficients.unsqueeze(1) * time_powers).sum(dim=-1)
        else:
            derivative_factors = torch.prod(torch.arange(1, polynomial_order + 1, dtype=dtype, device=device).unsqueeze(0) - 
                                            torch.arange(derivative_order, dtype=dtype, device=device).unsqueeze(1), dim=0)
            derivative_factors = torch.cat([torch.zeros(derivative_order, dtype=dtype, device=device), derivative_factors])
            values = (polynomial_coefficients.unsqueeze(1) * derivative_factors * time_powers[:, :, :-derivative_order]).sum(dim=-1)
        
        return values

    @staticmethod
    def evaluate_polynomials_vectorized(polynomial_coefficients, time_stamps, times, derivative_order):
        num_segments = polynomial_coefficients.shape[2]
        
        # Create a mask for each segment
        segment_masks = (times.unsqueeze(-1) >= time_stamps[:, :-1].unsqueeze(1)) & (times.unsqueeze(-1) < time_stamps[:, 1:].unsqueeze(1))
        segment_masks[:, :, -1] |= (times >= time_stamps[:, -1].unsqueeze(1))  # Include points at or after the last time stamp in the last segment
        
        # Evaluate polynomials for all segments
        all_values = torch.stack([TrajectoryUtils.evaluate_polynomial_vectorized(polynomial_coefficients[:, :, i], times, derivative_order) 
                                  for i in range(num_segments)], dim=-1)
        
        # Use the masks to select the correct values
        values = (all_values * segment_masks.float()).sum(dim=-1)
        
        return values

    
class UAVTrajectoryPlanner(nn.Module):
    def __init__(self, total_time, poly_order, start_vel, start_acc, end_vel, end_acc):
        super(UAVTrajectoryPlanner, self).__init__()
        self.total_time = total_time
        self.poly_order = poly_order
        self.start_vel = start_vel.to(device=device, dtype=dtype)
        self.start_acc = start_acc.to(device=device, dtype=dtype)
        self.end_vel = end_vel.to(device=device, dtype=dtype)
        self.end_acc = end_acc.to(device=device, dtype=dtype)
        self.waypoints = None
        self.qp_solver = None
        self.dtype=torch.float64
        self.device='cuda'
        
    def init_time_segments(self, waypoints, total_time):
        if waypoints.shape[1] == 2:
            return torch.tensor([0, total_time], dtype=dtype, device=device)
        num_segments = waypoints.shape[1] - 1
        time_intervals = torch.linspace(0, self.total_time, num_segments + 1, 
                                        dtype=dtype, 
                                        device=device)
        return time_intervals

    def init_time_segments1(self, waypoints, total_time):
        if waypoints.shape[1] == 2:
            return torch.tensor([0, total_time], dtype=dtype, device=device)
        num_waypoints = waypoints.shape[1]
        num_segments = num_waypoints - 1
        
        equal_segments = torch.full((num_segments,), total_time / num_segments, 
                                    dtype=dtype, device=device)
        
        differences = waypoints[:, 1:] - waypoints[:, :-1]
        distances = torch.sqrt(torch.sum(differences ** 2, dim=0) + 1e-10)  # Add small epsilon
        
        total_distance = torch.sum(distances)
        if total_distance == 0:
            return torch.linspace(0, total_time, num_waypoints, dtype=self.dtype, device=self.device)
        
        normalized_distances = distances / total_distance
        
        epsilon = 1e-3
        combined_segments = (1 - epsilon) * equal_segments + epsilon * normalized_distances * total_time
        combined_segments = combined_segments * (total_time / combined_segments.sum())
        
        time_intervals = torch.zeros(num_waypoints, dtype=dtype, device=device)
        time_intervals[1:] = torch.cumsum(combined_segments, dim=0)
        
        return time_intervals

    def solve_minimum_snap(self, waypoints, time_stamps, poly_order, start_vel, start_acc, end_vel, end_acc):
        start_pos = waypoints[0]
        end_pos = waypoints[-1]
        num_segments = len(waypoints) - 1
        num_coefficients = poly_order + 1

        Q_all = TrajectoryUtils.compute_Q_matrix(poly_order, 3, time_stamps)
        b_all = torch.zeros(Q_all.shape[0], dtype=dtype, device=device)

        Aeq = torch.zeros(4 * num_segments + 2, num_coefficients * num_segments, dtype=dtype, device=device)
        beq = torch.zeros(4 * num_segments + 2, dtype=dtype, device=device)

        # Initial and final conditions
        Aeq[0:3, :num_coefficients] = torch.stack([
            TrajectoryUtils.calculate_time_vector_vectorized(time_stamps[0].unsqueeze(0), poly_order, 0)[0],
            TrajectoryUtils.calculate_time_vector_vectorized(time_stamps[0].unsqueeze(0), poly_order, 1)[0],
            TrajectoryUtils.calculate_time_vector_vectorized(time_stamps[0].unsqueeze(0), poly_order, 2)[0]
        ])
        Aeq[3:6, -num_coefficients:] = torch.stack([
            TrajectoryUtils.calculate_time_vector_vectorized(time_stamps[-1].unsqueeze(0), poly_order, 0)[0],
            TrajectoryUtils.calculate_time_vector_vectorized(time_stamps[-1].unsqueeze(0), poly_order, 1)[0],
            TrajectoryUtils.calculate_time_vector_vectorized(time_stamps[-1].unsqueeze(0), poly_order, 2)[0]
        ])
        beq[0:6] = torch.tensor([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc], dtype=dtype, device=device)

        # Waypoint constraints
        waypoint_times = time_stamps[1:-1]
        waypoint_vectors = TrajectoryUtils.calculate_time_vector_vectorized(waypoint_times, poly_order, 0)
        for i in range(num_segments - 1):
            Aeq[6+i, (i+1)*num_coefficients:(i+2)*num_coefficients] = waypoint_vectors[i]
        beq[6:6+num_segments-1] = waypoints[1:-1]

        # Continuity constraints
        continuity_times = time_stamps[1:-1]
        continuity_vectors = TrajectoryUtils.calculate_time_vector_vectorized(continuity_times, poly_order, 0)
        continuity_vectors_v = TrajectoryUtils.calculate_time_vector_vectorized(continuity_times, poly_order, 1)
        continuity_vectors_a = TrajectoryUtils.calculate_time_vector_vectorized(continuity_times, poly_order, 2)

        for i in range(num_segments - 1):
            row_start = 6 + num_segments - 1 + 3 * i
            col_start = i * num_coefficients
            Aeq[row_start, col_start:col_start+2*num_coefficients] = torch.cat([continuity_vectors[i], -continuity_vectors[i]])
            Aeq[row_start+1, col_start:col_start+2*num_coefficients] = torch.cat([continuity_vectors_v[i], -continuity_vectors_v[i]])
            Aeq[row_start+2, col_start:col_start+2*num_coefficients] = torch.cat([continuity_vectors_a[i], -continuity_vectors_a[i]])
        

        

        G_dummy = torch.zeros(1, Q_all.size(0), Q_all.size(0), dtype=dtype, device=device)
        h_dummy = torch.zeros(1, Q_all.size(0), dtype=dtype, device=device)
    
        
        lambda_val = 1e-5  # Regularization parameter
        initial_pos = 0  # Correct index for final position
        for i in range(num_coefficients):
            Q_all[initial_pos + i, initial_pos + i] += lambda_val
            b_all[initial_pos + i] -= lambda_val * end_pos
        final_position_index = (num_segments - 1) * num_coefficients  # Correct index for final position
        for i in range(num_coefficients):
            Q_all[final_position_index + i, final_position_index + i] += lambda_val
            b_all[final_position_index + i] -= lambda_val * end_pos
        
        Q_all_updated = Q_all+torch.eye(Q_all.size(0), dtype=dtype, device=device) * 1e-4

        # Q_all_updated = TrajectoryUtils.ensure_spd_matrix(Q_all_updated, method='cholesky_with_perturbation')
        eigenvalues = torch.linalg.eigvalsh(Q_all_updated)
        if torch.any(eigenvalues <= 0):
            print("Q is not positive definite")

        solver_options = {'eps': 1e-24, 'maxIter': 50, 'solver': qpth.qp.QPSolvers.PDIPM_BATCHED}
    
        try:
            solution = QPFunction(verbose=-1, **solver_options)(
               Q_all_updated, b_all, G_dummy, h_dummy, Aeq, beq
            )
            # solution = solution.requires_grad_()
            # print(f"solution requires_grad: {solution.requires_grad}")
        except RuntimeError as e:
            print(f"QPFunction failed with error: {e}")
            solution = torch.zeros(num_segments * num_coefficients, dtype=self.dtype, device=self.device)

        polynomial_coefficients = solution.view(num_segments, num_coefficients).transpose(0, 1)
        # polynomial_coefficients.retain_grad()
        
        if torch.isnan(polynomial_coefficients).any():
            print("NaNs detected in solve_minimum_snap polynomial_coefficients")
        
        return polynomial_coefficients

    def forward(self, waypoints, total_time):
        waypoints = waypoints.to(device=device, dtype=dtype)
        self.waypoints = waypoints
        optimized_time_segments = self.init_time_segments(waypoints, total_time)
        
        if torch.isnan(optimized_time_segments).any():
            print("NaNs detected in forward optimized_time_segments")
        
        polys_x = self.solve_minimum_snap(waypoints[0], optimized_time_segments, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
        
        polys_y = self.solve_minimum_snap(waypoints[1], optimized_time_segments, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        
        polys_z = self.solve_minimum_snap(waypoints[2], optimized_time_segments, self.poly_order, self.start_vel[2], self.start_acc[2], self.end_vel[2], self.end_acc[2])
        
        return polys_x, polys_y, polys_z, optimized_time_segments



