# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the dtype and device at one place
# dtype = torch.float64
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class MinimumSnapTrajectoryPlanner:
#     def __init__(self, total_time, poly_order, start_vel, start_acc, end_vel, end_acc):
#         self.total_time = total_time
#         self.poly_order = poly_order
#         self.start_vel = start_vel.to(device)
#         self.start_acc = start_acc.to(device)
#         self.end_vel = end_vel.to(device)
#         self.end_acc = end_acc.to(device)

#     def forward(self, waypoints, total_time):
#         waypoints = waypoints.to(device)
#         ts = self.arrange_T(waypoints)
#         polys_x = self.minimum_snap_single_axis_close_form(waypoints[0], ts, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
#         polys_y = self.minimum_snap_single_axis_close_form(waypoints[1], ts, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
#         polys_z = self.minimum_snap_single_axis_close_form(waypoints[2], ts, self.poly_order, self.start_vel[2], self.start_acc[2], self.end_vel[2], self.end_acc[2])
#         return polys_x, polys_y, polys_z, ts

#     def arrange_T(self, waypoints):
#         """
#         Arranges the waypoints in time.

#         Args:
#             waypoints (Tensor): Tensor of waypoints

#         Returns:
#             Tensor: Arranged time tensor
#         """
#         # differences = waypoints[:, 1:] - waypoints[:, :-1]
#         # distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
#         # time_fraction = self.total_time / torch.sum(distances)
#         # arranged_time = torch.cat([torch.tensor([0], device=device), torch.cumsum(distances * time_fraction, dim=0)])
#         if waypoints.shape[1] == 2:
#             return torch.tensor([0, self.total_time], dtype=waypoints.dtype, device=waypoints.device)

#         # Existing code for more than two waypoints
#         # differences = waypoints[:, 1:] - waypoints[:, :-1]
#         # distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
#         # # Prevent division by zero by ensuring all distances have a minimum value
#         # safe_distances = torch.clamp(distances, min=1e-6)
#         # time_fraction = self.total_time / torch.sum(safe_distances)
#         # arranged_time = torch.cumsum(safe_distances * time_fraction, dim=0)
#         # arranged_time = torch.cat([torch.tensor([0], dtype=waypoints.dtype, device=waypoints.device), arranged_time])
        
#         # if torch.isnan(arranged_time).any():
#         #     print("NaNs detected in arranged_time:", arranged_time)
#         num_segments = waypoints.shape[1] - 1
#         time_intervals = torch.linspace(0, self.total_time, num_segments + 1, 
#                                     dtype=waypoints.dtype, 
#                                     device=waypoints.device)
            
#         return time_intervals

#     def compute_Q_matrix(self, poly_order, derivative_order, start_time, end_time):
#         """
#         Computes the Q matrix for the minimum snap problem.

#         Args:
#             poly_order (int): Order of the polynomial
#             derivative_order (int): Derivative order
#             start_time (float): Start time
#             end_time (float): End time

#         Returns:
#             Tensor: Q matrix
#         """
#         time_diff_powers = torch.zeros((poly_order - derivative_order) * 2 + 1, dtype=dtype, device=device)
#         for i in range((poly_order - derivative_order) * 2 + 1):
#             time_diff_powers[i] = end_time ** (i + 1) - start_time ** (i + 1)

#         Q_matrix = torch.zeros(poly_order + 1, poly_order + 1, dtype=dtype, device=device)
#         for i in range(derivative_order + 1, poly_order + 2):
#             for j in range(i, poly_order + 2):
#                 k1 = i - derivative_order - 1
#                 k2 = j - derivative_order - 1
#                 k = k1 + k2 + 1
#                 prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + derivative_order + 1), dtype=dtype, device=device))
#                 prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + derivative_order + 1), dtype=dtype, device=device))
#                 Q_matrix[i - 1, j - 1] = prod_k1 * prod_k2 / k * time_diff_powers[k - 1]
#                 Q_matrix[j - 1, i - 1] = Q_matrix[i - 1, j - 1]

#         return Q_matrix

#     def compute_M(self, n_poly, n_continuous):
#         """
#         Computes the M matrix for the minimum snap problem.

#         Args:
#             n_poly (int): Number of polynomials
#             n_continuous (int): Number of continuous derivatives

#         Returns:
#             Tensor: M matrix
#         """
#         num_d = n_continuous * (n_poly + 1)
#         M = torch.zeros(n_poly * 2 * n_continuous, num_d, dtype=dtype, device=device)
        
#         for i in range(n_poly):
#             start_row = i * 2 * n_continuous
#             start_col = i * n_continuous
            
#             # First set of rows for each polynomial
#             M[start_row:start_row + n_continuous, start_col:start_col + n_continuous] = torch.eye(n_continuous, dtype=dtype, device=device)
            
#             # Second set of rows for each polynomial
#             M[start_row + n_continuous:start_row + 2*n_continuous, start_col + n_continuous:start_col + 2*n_continuous] = torch.eye(n_continuous, dtype=dtype, device=device)
        
#         return M

#     def minimum_snap_single_axis_close_form(self, wayp, ts, n_order, v0, a0, v1, a1):
#         """
#         Solves the minimum snap trajectory for a single axis using a closed-form solution.

#         Args:
#             wayp (Tensor): Waypoints for the axis
#             ts (Tensor): Time stamps for waypoints
#             n_order (int): Order of the polynomial
#             v0 (float): Initial velocity
#             a0 (float): Initial acceleration
#             v1 (float): Final velocity
#             a1 (float): Final acceleration

#         Returns:
#             Tensor: Polynomial coefficients for the axis
#         """
#         n_coef = n_order + 1
#         n_poly = len(wayp) - 1

#         # Ensure ts is a 1-dimensional tensor
#         ts = ts.view(-1)

#         # Compute Q matrix
#         Q_all = torch.block_diag(*[self.compute_Q_matrix(n_order, 3, ts[i].item(), ts[i + 1].item()) for i in range(n_poly)])
#         if torch.isnan(Q_all).any():
#             print("NaNs detected in Q_all:", Q_all)

#         # Compute Tk matrix
#         tk = torch.zeros(n_poly + 1, n_coef, dtype=dtype, device=device)
#         for i in range(n_coef):
#             tk[:, i] = ts.pow(i)
#         if torch.isnan(tk).any():
#             print("NaNs detected in tk:", tk)

#         # Compute A matrix
#         n_continuous = 3
#         A = torch.zeros(n_continuous * 2 * n_poly, n_coef * n_poly, dtype=dtype, device=device)
#         for i in range(n_poly):
#             for j in range(n_continuous):
#                 for k in range(j, n_coef):
#                     t1 = tk[i, k-j] if k != j else 1
#                     t2 = tk[i+1, k-j] if k != j else 1
#                     A[n_continuous*2*i+j, n_coef*i+k] = torch.prod(torch.arange(k-j+1, k+1, dtype=dtype, device=device)) * t1
#                     A[n_continuous*2*i+n_continuous+j, n_coef*i+k] = torch.prod(torch.arange(k-j+1, k+1, dtype=dtype, device=device)) * t2
#         if torch.isnan(A).any():
#             print("NaNs detected in A:", A)

#         # Compute M matrix
#         M = self.compute_M(n_poly, n_continuous)
#         if torch.isnan(M).any():
#             print("NaNs detected in M:", M)

#         # Compute C matrix
#         num_d = n_continuous * (n_poly + 1)
#         C = torch.eye(num_d, dtype=dtype, device=device)
#         df = torch.cat([wayp, torch.tensor([v0, a0, v1, a1], dtype=dtype, device=device)])
#         fix_idx = torch.cat([torch.arange(0, num_d, 3, dtype=torch.int64, device=device), torch.tensor([1, 2, num_d-2, num_d-1], dtype=torch.int64, device=device)])
#         free_idx = torch.tensor([i for i in range(num_d) if i not in fix_idx], dtype=torch.int64, device=device)
#         C = torch.cat([C[:, fix_idx], C[:, free_idx]], dim=1)
#         if torch.isnan(C).any():
#             print("NaNs detected in C:", C)

#         # Solve for polynomial coefficients
#         AiMC = torch.linalg.inv(A) @ M @ C
#         if torch.isnan(AiMC).any():
#             print("NaNs detected in AiMC:", AiMC)
#         R = AiMC.T @ Q_all @ AiMC
#         if torch.isnan(R).any():
#             print("NaNs detected in R:", R)
#         n_fix = len(fix_idx)
#         Rff = R[:n_fix, :n_fix]
#         Rpp = R[n_fix:, n_fix:]
#         Rfp = R[:n_fix, n_fix:]
#         if torch.isnan(Rff).any():
#             print("NaNs detected in Rff:", Rff)
#         if torch.isnan(Rpp).any():
#             print("NaNs detected in Rpp:", Rpp)
#         if torch.isnan(Rfp).any():
#             print("NaNs detected in Rfp:", Rfp)

#         dp = -torch.linalg.inv(Rpp) @ Rfp.T @ df
#         if torch.isnan(dp).any():
#             print("NaNs detected in dp:", dp)
#         p = AiMC @ torch.cat([df, dp])
#         if torch.isnan(p).any():
#             print("NaNs detected in p:", p)
#         p = p.reshape(-1, 6).T
#         if torch.isnan(p).any():
#             print("NaNs detected in final polynomial coefficients:", p)

#         return p

#     def evaluate_polynomial_vectorized(self, polynomial_coefficients, times, derivative_order):
#         device = times.device
#         polynomial_order = polynomial_coefficients.shape[1] - 1
#         powers = torch.arange(polynomial_order + 1, device=device).unsqueeze(0).unsqueeze(0)
#         time_powers = times.unsqueeze(-1).pow(powers)
        
#         if derivative_order <= 0:
#             values = (polynomial_coefficients.unsqueeze(1) * time_powers).sum(dim=-1)
#         else:
#             derivative_factors = torch.prod(torch.arange(1, polynomial_order + 1, device=device).unsqueeze(0) - 
#                                             torch.arange(derivative_order, device=device).unsqueeze(1), dim=0)
#             derivative_factors = torch.cat([torch.zeros(derivative_order, device=device), derivative_factors])
#             values = (polynomial_coefficients.unsqueeze(1) * derivative_factors * time_powers[:, :, :-derivative_order]).sum(dim=-1)
        
#         return values

#     def evaluate_polynomials_vectorized(self, polynomial_coefficients, time_stamps, times, derivative_order):
#         device = times.device
#         num_segments = polynomial_coefficients.shape[2]
        
#         # Create a mask for each segment
#         segment_masks = (times.unsqueeze(-1) >= time_stamps[:, :-1].unsqueeze(1)) & (times.unsqueeze(-1) < time_stamps[:, 1:].unsqueeze(1))
#         segment_masks[:, :, -1] |= (times >= time_stamps[:, -1].unsqueeze(1))  # Include points at or after the last time stamp in the last segment
        
#         # Evaluate polynomials for all segments
#         all_values = torch.stack([self.evaluate_polynomial_vectorized(polynomial_coefficients[:, :, i], times, derivative_order) 
#                                   for i in range(num_segments)], dim=-1)
        
#         # Use the masks to select the correct values
#         values = (all_values * segment_masks.float()).sum(dim=-1)
        
#         return values
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the dtype and device at one place
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinimumSnapTrajectoryPlanner:
    def __init__(self, total_time, poly_order, start_vel, start_acc, end_vel, end_acc):
        self.total_time = total_time
        self.poly_order = poly_order
        self.start_vel = start_vel.to(device)
        self.start_acc = start_acc.to(device)
        self.end_vel = end_vel.to(device)
        self.end_acc = end_acc.to(device)

    def forward(self, waypoints, total_time):
        waypoints = waypoints.to(device)
        ts = self.arrange_T(waypoints)
        polys_x = self.minimum_snap_single_axis_close_form(waypoints[0], ts, self.poly_order, self.start_vel[0], self.start_acc[0], self.end_vel[0], self.end_acc[0])
        polys_y = self.minimum_snap_single_axis_close_form(waypoints[1], ts, self.poly_order, self.start_vel[1], self.start_acc[1], self.end_vel[1], self.end_acc[1])
        polys_z = self.minimum_snap_single_axis_close_form(waypoints[2], ts, self.poly_order, self.start_vel[2], self.start_acc[2], self.end_vel[2], self.end_acc[2])
        return polys_x, polys_y, polys_z, ts

    def arrange_T(self, waypoints):
        """
        Arranges the waypoints in time.

        Args:
            waypoints (Tensor): Tensor of waypoints

        Returns:
            Tensor: Arranged time tensor
        """
        # differences = waypoints[:, 1:] - waypoints[:, :-1]
        # distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
        # time_fraction = self.total_time / torch.sum(distances)
        # arranged_time = torch.cat([torch.tensor([0], device=device), torch.cumsum(distances * time_fraction, dim=0)])
        if waypoints.shape[1] == 2:
            return torch.tensor([0, self.total_time], dtype=waypoints.dtype, device=waypoints.device)

        # Existing code for more than two waypoints
        # differences = waypoints[:, 1:] - waypoints[:, :-1]
        # distances = torch.sqrt(torch.sum(differences ** 2, dim=0))
        # # Prevent division by zero by ensuring all distances have a minimum value
        # safe_distances = torch.clamp(distances, min=1e-6)
        # time_fraction = self.total_time / torch.sum(safe_distances)
        # arranged_time = torch.cumsum(safe_distances * time_fraction, dim=0)
        # arranged_time = torch.cat([torch.tensor([0], dtype=waypoints.dtype, device=waypoints.device), arranged_time])
        
        # if torch.isnan(arranged_time).any():
        #     print("NaNs detected in arranged_time:", arranged_time)
        num_segments = waypoints.shape[1] - 1
        time_intervals = torch.linspace(0, self.total_time, num_segments + 1, 
                                    dtype=waypoints.dtype, 
                                    device=waypoints.device)
            
        return time_intervals

    def compute_Q_matrix(self, poly_order, derivative_order, start_time, end_time):
        """
        Computes the Q matrix for the minimum snap problem.

        Args:
            poly_order (int): Order of the polynomial
            derivative_order (int): Derivative order
            start_time (float): Start time
            end_time (float): End time

        Returns:
            Tensor: Q matrix
        """
        time_diff_powers = torch.zeros((poly_order - derivative_order) * 2 + 1, dtype=dtype, device=device)
        for i in range((poly_order - derivative_order) * 2 + 1):
            time_diff_powers[i] = end_time ** (i + 1) - start_time ** (i + 1)

        Q_matrix = torch.zeros(poly_order + 1, poly_order + 1, dtype=dtype, device=device)
        for i in range(derivative_order + 1, poly_order + 2):
            for j in range(i, poly_order + 2):
                k1 = i - derivative_order - 1
                k2 = j - derivative_order - 1
                k = k1 + k2 + 1
                prod_k1 = torch.prod(torch.tensor(range(k1 + 1, k1 + derivative_order + 1), dtype=dtype, device=device))
                prod_k2 = torch.prod(torch.tensor(range(k2 + 1, k2 + derivative_order + 1), dtype=dtype, device=device))
                Q_matrix[i - 1, j - 1] = prod_k1 * prod_k2 / k * time_diff_powers[k - 1]
                Q_matrix[j - 1, i - 1] = Q_matrix[i - 1, j - 1]

        return Q_matrix

    def compute_M(self, n_poly, n_continuous):
        """
        Computes the M matrix for the minimum snap problem.

        Args:
            n_poly (int): Number of polynomials
            n_continuous (int): Number of continuous derivatives

        Returns:
            Tensor: M matrix
        """
        num_d = n_continuous * (n_poly + 1)
        M = torch.zeros(n_poly * 2 * n_continuous, num_d, dtype=dtype, device=device)
        
        for i in range(n_poly):
            start_row = i * 2 * n_continuous
            start_col = i * n_continuous
            
            # First set of rows for each polynomial
            M[start_row:start_row + n_continuous, start_col:start_col + n_continuous] = torch.eye(n_continuous, dtype=dtype, device=device)
            
            # Second set of rows for each polynomial
            M[start_row + n_continuous:start_row + 2*n_continuous, start_col + n_continuous:start_col + 2*n_continuous] = torch.eye(n_continuous, dtype=dtype, device=device)
        
        return M

    def minimum_snap_single_axis_close_form(self, wayp, ts, n_order, v0, a0, v1, a1):
        """
        Solves the minimum snap trajectory for a single axis using a closed-form solution.

        Args:
            wayp (Tensor): Waypoints for the axis
            ts (Tensor): Time stamps for waypoints
            n_order (int): Order of the polynomial
            v0 (float): Initial velocity
            a0 (float): Initial acceleration
            v1 (float): Final velocity
            a1 (float): Final acceleration

        Returns:
            Tensor: Polynomial coefficients for the axis
        """
        n_coef = n_order + 1
        n_poly = len(wayp) - 1

        # Ensure ts is a 1-dimensional tensor
        ts = ts.view(-1)

        # Compute Q matrix
        Q_all = torch.block_diag(*[self.compute_Q_matrix(n_order, 3, ts[i].item(), ts[i + 1].item()) for i in range(n_poly)])
        if torch.isnan(Q_all).any():
            print("NaNs detected in Q_all:", Q_all)

        # Compute Tk matrix
        tk = torch.zeros(n_poly + 1, n_coef, dtype=dtype, device=device)
        for i in range(n_coef):
            tk[:, i] = ts.pow(i)
        if torch.isnan(tk).any():
            print("NaNs detected in tk:", tk)

        # Compute A matrix
        n_continuous = 3
        A = torch.zeros(n_continuous * 2 * n_poly, n_coef * n_poly, dtype=dtype, device=device)
        for i in range(n_poly):
            for j in range(n_continuous):
                for k in range(j, n_coef):
                    t1 = tk[i, k-j] if k != j else 1
                    t2 = tk[i+1, k-j] if k != j else 1
                    A[n_continuous*2*i+j, n_coef*i+k] = torch.prod(torch.arange(k-j+1, k+1, dtype=dtype, device=device)) * t1
                    A[n_continuous*2*i+n_continuous+j, n_coef*i+k] = torch.prod(torch.arange(k-j+1, k+1, dtype=dtype, device=device)) * t2
        if torch.isnan(A).any():
            print("NaNs detected in A:", A)

        # Compute M matrix
        M = self.compute_M(n_poly, n_continuous)
        if torch.isnan(M).any():
            print("NaNs detected in M:", M)

        # Compute C matrix
        num_d = n_continuous * (n_poly + 1)
        C = torch.eye(num_d, dtype=dtype, device=device)
        df = torch.cat([wayp, torch.tensor([v0, a0, v1, a1], dtype=dtype, device=device)])
        fix_idx = torch.cat([torch.arange(0, num_d, 3, dtype=torch.int64, device=device), torch.tensor([1, 2, num_d-2, num_d-1], dtype=torch.int64, device=device)])
        free_idx = torch.tensor([i for i in range(num_d) if i not in fix_idx], dtype=torch.int64, device=device)
        C = torch.cat([C[:, fix_idx], C[:, free_idx]], dim=1)
        if torch.isnan(C).any():
            print("NaNs detected in C:", C)

        # Solve for polynomial coefficients
        AiMC = torch.linalg.inv(A) @ M @ C
        if torch.isnan(AiMC).any():
            print("NaNs detected in AiMC:", AiMC)
        R = AiMC.T @ Q_all @ AiMC
        if torch.isnan(R).any():
            print("NaNs detected in R:", R)
        n_fix = len(fix_idx)
        Rff = R[:n_fix, :n_fix]
        Rpp = R[n_fix:, n_fix:]
        Rfp = R[:n_fix, n_fix:]
        if torch.isnan(Rff).any():
            print("NaNs detected in Rff:", Rff)
        if torch.isnan(Rpp).any():
            print("NaNs detected in Rpp:", Rpp)
        if torch.isnan(Rfp).any():
            print("NaNs detected in Rfp:", Rfp)

        dp = -torch.linalg.inv(Rpp) @ Rfp.T @ df
        if torch.isnan(dp).any():
            print("NaNs detected in dp:", dp)
        p = AiMC @ torch.cat([df, dp])
        if torch.isnan(p).any():
            print("NaNs detected in p:", p)
        p = p.reshape(-1, 6).T
        if torch.isnan(p).any():
            print("NaNs detected in final polynomial coefficients:", p)

        return p

    def evaluate_polynomial_vectorized(self, polynomial_coefficients, times, derivative_order):
        device = times.device
        polynomial_order = polynomial_coefficients.shape[1] - 1
        powers = torch.arange(polynomial_order + 1, device=device).unsqueeze(0).unsqueeze(0)
        time_powers = times.unsqueeze(-1).pow(powers)
        # print(f"polynomial_coefficients shape: {polynomial_coefficients.shape}")
        # print(f"times shape: {times.shape}")
        # print(f"derivative_order: {derivative_order}")
        # print(f"polynomial_order: {polynomial_order}")
        
        if derivative_order <= 0:
            values = (polynomial_coefficients.unsqueeze(1) * time_powers).sum(dim=-1)
        # else:
        #     derivative_factors = torch.prod(torch.arange(1, polynomial_order + 1, device=device).unsqueeze(0) - 
        #                                     torch.arange(derivative_order, device=device).unsqueeze(1), dim=0)
        #     derivative_factors = torch.cat([torch.zeros(derivative_order, device=device), derivative_factors])
        #     # values = (polynomial_coefficients.unsqueeze(1) * derivative_factors * time_powers[:, :, :-derivative_order]).sum(dim=-1)
        #     print(f"derivative_factors shape: {derivative_factors.shape}")
        #     print(f"time_powers shape: {time_powers.shape}")
        #     print(f"time_powers[:, :, :-derivative_order] shape: {time_powers[:, :, :-derivative_order].shape}")

        #     # derivative_factors = derivative_factors[:-derivative_order]

        #     # values = (polynomial_coefficients.unsqueeze(1) * derivative_factors * time_powers[:, :, :-derivative_order]).sum(dim=-1)
        #      # Adjust both polynomial_coefficients and derivative_factors
        #     adjusted_coeffs = polynomial_coefficients[:, derivative_order:]
        #     adjusted_factors = derivative_factors[derivative_order:]
            
        #     values = (adjusted_coeffs.unsqueeze(1) * adjusted_factors * time_powers[:, :, :-derivative_order]).sum(dim=-1)
        
        # print(f"values shape: {values.shape}")
        else:
            # Calculate derivative factors
            derivative_factors = torch.ones(polynomial_order + 1, dtype=dtype, device=device)
            for i in range(1, derivative_order + 1):
                derivative_factors[i:] *= torch.arange(i, polynomial_order + 1, dtype=dtype, device=device)
            
            # Adjust coefficients and time powers
            adjusted_coeffs = polynomial_coefficients[:, derivative_order:]
            adjusted_time_powers = time_powers[:, :, :-derivative_order]
            
            # print(f"derivative_factors shape: {derivative_factors.shape}")
            # print(f"adjusted_coeffs shape: {adjusted_coeffs.shape}")
            # print(f"adjusted_time_powers shape: {adjusted_time_powers.shape}")
            
            values = (adjusted_coeffs.unsqueeze(1) * derivative_factors[derivative_order:] * adjusted_time_powers).sum(dim=-1)
        
        # print(f"values shape: {values.shape}")
    # return values
       
        
        return values

    def evaluate_polynomials_vectorized(self, polynomial_coefficients, time_stamps, times, derivative_order):
        device = times.device
        num_segments = polynomial_coefficients.shape[2]
        # print(f"polynomial_coefficients shape: {polynomial_coefficients.shape}")
        # print(f"time_stamps shape: {time_stamps.shape}")
        # print(f"times shape: {times.shape}")
        # print(f"derivative_order: {derivative_order}")
        # print(f"num_segments: {num_segments}")
        
        # Create a mask for each segment
        segment_masks = (times.unsqueeze(-1) >= time_stamps[:, :-1].unsqueeze(1)) & (times.unsqueeze(-1) < time_stamps[:, 1:].unsqueeze(1))
        segment_masks[:, :, -1] |= (times >= time_stamps[:, -1].unsqueeze(1))  # Include points at or after the last time stamp in the last segment
        
        # Evaluate polynomials for all segments
        # all_values = torch.stack([self.evaluate_polynomial_vectorized(polynomial_coefficients[:, :, i], times, derivative_order) 
                                #   for i in range(num_segments)], dim=-1)

        # Evaluate polynomials for all segments
        all_values = []
        for i in range(num_segments):
            segment_coeffs = polynomial_coefficients[:, :, i]
            segment_values = self.evaluate_polynomial_vectorized(segment_coeffs, times, derivative_order)
            all_values.append(segment_values)
        all_values = torch.stack(all_values, dim=-1)
        
        # print(f"all_values shape: {all_values.shape}")
        # print(f"segment_masks shape: {segment_masks.shape}")

        # print(f"all_values shape: {all_values.shape}")
        # print(f"segment_masks shape: {segment_masks.shape}")
        
        # Use the masks to select the correct values
        values = (all_values * segment_masks.float()).sum(dim=-1)
        
        return values
