import numpy as np
import matplotlib.pyplot as plt

import torch
import gpytorch

# Helper class
# ===============================
class SecondOrderAgent:
    """
    A point mass agent with second order dynamics.
    """
    def __init__(
        self,
        x,
        nbDataPoints,
        max_dx=1,
        max_ddx=0.2,
        dt=1,
    ):
        self.x = np.array(x)  # position
        # determine which dimesnion we are in from given position
        self.nbVarX = len(x)
        self.dx = np.zeros(self.nbVarX)  # velocity

        self.t = 0  # time
        self.dt = dt  # time step
        self.nbDatapoints = nbDataPoints

        self.max_dx = max_dx
        self.max_ddx = max_ddx

        # we will store the actual and desired position
        # of the agent over the timesteps
        self.x_arr = np.zeros((self.nbDatapoints, self.nbVarX))
        self.des_x_arr = np.zeros((self.nbDatapoints, self.nbVarX))

    def update(self, gradient):
        """
        set the acceleration of the agent to clamped gradient
        compute the position at t+1 based on clamped acceleration
        and velocity
        """
        ddx = gradient # we use gradient of the potential field as acceleration
        # clamp acceleration if needed
        if np.linalg.norm(ddx) > self.max_ddx:
            ddx = self.max_ddx * ddx / np.linalg.norm(ddx)

        self.x = self.x + self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        self.x_arr[self.t] = np.copy(self.x)
        self.t += 1

        self.dx += self.dt * ddx  # compute the velocity
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)

# Helper functions for HEDAC
# ===============================
def rbf(mean, x, eps):
    """
    Radial basis function w/ Gaussian Kernel
    """
    d = x - mean  # radial distance
    l2_norm_squared = np.dot(d, d)
    # eps is the shape parameter that can be interpreted as the inverse of the radius
    return np.exp(-eps * l2_norm_squared)


def normalize_mat(mat):
    return mat / (np.sum(mat) + 1e-10)


def calculate_gradient(agent, gradient_x, gradient_y, param):
    """
    Calculate movement direction of the agent by considering the gradient
    of the temperature field near the agent
    """
    # find agent pos on the grid as integer indices
    adjusted_position = agent.x / param.dx
    # note x axis corresponds to col and y axis corresponds to row
    col, row = adjusted_position.astype(int)


    gradient = np.zeros(2)
    # if agent is inside the grid, interpolate the gradient for agent position
    if row > 0 and row < param.height - 1 and col > 0 and col < param.width - 1:
        gradient[0] = bilinear_interpolation(gradient_x, adjusted_position)
        gradient[1] = bilinear_interpolation(gradient_y, adjusted_position)

    # if kernel around the agent is outside the grid,
    # use the gradient to direct the agent inside the grid
    boundary_gradient = 2  # 0.1
    pad = param.kernel_size - 1
    if row <= pad:
        gradient[1] = boundary_gradient
    elif row >= param.height - 1 - pad:
        gradient[1] = -boundary_gradient

    if col <= pad:
        gradient[0] = boundary_gradient
    elif col >= param.width - pad:
        gradient[0] = -boundary_gradient

    return gradient


def clamp_kernel_1d(x, low_lim, high_lim, kernel_size):
    """
    A function to calculate the start and end indices
    of the kernel around the agent that is inside the grid
    i.e. clamp the kernel by the grid boundaries
    """
    start_kernel = low_lim
    start_grid = x - (kernel_size // 2)
    num_kernel = kernel_size
    # bound the agent to be inside the grid
    if x <= -(kernel_size // 2):
        x = -(kernel_size // 2) + 1
    elif x >= high_lim + (kernel_size // 2):
        x = high_lim + (kernel_size // 2) - 1

    # if agent kernel around the agent is outside the grid,
    # clamp the kernel by the grid boundaries
    if start_grid < low_lim:
        start_kernel = kernel_size // 2 - x - 1
        num_kernel = kernel_size - start_kernel - 1
        start_grid = low_lim
    elif start_grid + kernel_size >= high_lim:
        num_kernel -= x - (high_lim - num_kernel // 2 - 1)
    if num_kernel > low_lim:
        grid_indices = slice(start_grid, start_grid + num_kernel)

    return grid_indices, start_kernel, num_kernel


def agent_block(min_val, agent_radius):
    """
    A matrix representing the shape of an agent (e.g, RBF with Gaussian kernel). 
    min_val is the upper bound on the minimum value of the agent block.
    """
    nbVarX = 2  # number of dimensions of space

    eps = 1.0 / agent_radius  # shape parameter of the RBF
    l2_sqrd = (
        -np.log(min_val) / eps
    )  # squared maximum distance from the center of the agent block
    l2_sqrd_single = (
        l2_sqrd / nbVarX
    )  # maximum squared distance on a single axis since sum of all axes equal to l2_sqrd
    l2_single = np.sqrt(l2_sqrd_single)  # maximum distance on a single axis
    # round to the nearest larger integer
    if l2_single.is_integer(): 
        l2_upper = int(l2_single)
    else:
        l2_upper = int(l2_single) + 1
    # agent block is symmetric about the center
    num_rows = l2_upper * 2 + 1
    num_cols = num_rows
    block = np.zeros((num_rows, num_cols))
    center = np.array([num_rows // 2, num_cols // 2])
    for i in range(num_rows):
        for j in range(num_cols):
            block[i, j] = rbf(np.array([j, i]), center, eps)
    # we hope this value is close to zero 
    print(f"Minimum element of the block: {np.min(block)}" +
          " values smaller than this assumed as zero")
    return block


def offset(mat, i, j):
    """
    offset a 2D matrix by i, j
    """
    rows, cols = mat.shape
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]


def border_interpolate(x, length, border_type):
    """
    Helper function to interpolate border values based on the border type
    (gives the functionality of cv2.borderInterpolate function)
    """
    if border_type == "reflect101":
        if x < 0:
            return -x
        elif x >= length:
            return 2 * length - x - 2
    return x


def bilinear_interpolation(grid, pos):
    """
    Linear interpolating function on a 2-D grid
    """
    x, y = pos.astype(int)
    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1], "reflect101")
    x1 = border_interpolate(x + 1, grid.shape[1], "reflect101")
    y0 = border_interpolate(y, grid.shape[0], "reflect101")
    y1 = border_interpolate(y + 1, grid.shape[0], "reflect101")
    # Distance from lower integers
    xd = pos[0] - x0
    yd = pos[1] - y0
    # Interpolate on x-axis
    c01 = grid[y0, x0] * (1 - xd) + grid[y0, x1] * xd
    c11 = grid[y1, x0] * (1 - xd) + grid[y1, x1] * xd
    # Interpolate on y-axis
    c = c01 * (1 - yd) + c11 * yd
    return c


# Helper functions borrowed from SMC example given in 
# demo_ergodicControl_2D_01.py for using the same 
# target distribution and comparing the results
# of SMC and HEDAC
# ===============================
def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
    # Base case: A Hadamard matrix of size 1 is just [[1]].
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2.
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    # Combine the four sub-matrices to form a Hadamard matrix of size n.
    h = np.empty((n, n), dtype=int)
    h[:half_size,:half_size] = h_half
    h[half_size:,:half_size] = h_half
    h[:half_size:,half_size:] = h_half
    h[half_size:,half_size:] = -h_half

    return h


def get_GMM(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Gaussian centers
    Mu1 = [0.5, 0.7]
    Mu2 = [0.6, 0.3]
    # Gaussian covariances
    # direction vectors for constructing the covariance matrix using
    # outer product of a vector with itself then the principal direction
    # of covariance matrix becomes the given vector and its orthogonal
    # complement
    Sigma1_v = [0.3, 0.1]
    Sigma2_v = [0.1, 0.2]
    # scale
    Sigma1_scale = 5e-1
    Sigma2_scale = 3e-1
    # regularization
    Sigma1_regularization = np.eye(param.nbVarX) * 5e-3
    Sigma2_regularization = np.eye(param.nbVarX) * 1e-2
    # GMM Gaussian Mixture Model

    # Gaussian centers
    Mu = np.zeros((param.nbVarX, param.nbGaussian))
    Mu[:, 0] = np.array(Mu1)
    Mu[:, 1] = np.array(Mu2)
    # covariance matrices
    Sigma = np.zeros((param.nbVarX, param.nbVarX, param.nbGaussian))
    # construct the covariance matrix using the outer product
    Sigma[:, :, 0] = (
        np.vstack(Sigma1_v) @ np.vstack(Sigma1_v).T * Sigma1_scale
        + Sigma1_regularization
    )
    Sigma[:, :, 1] = (
        np.vstack(Sigma2_v) @ np.vstack(Sigma2_v).T * Sigma2_scale
        + Sigma2_regularization
    )
    # mixing. coefficients Priors (summing to one)
    Alpha = (
        np.ones(param.nbGaussian) / param.nbGaussian
    )
    return Mu, Sigma, Alpha

def get_fixed_GMM(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Gaussian centers
    Mu1 = [0.5, 0.7]
    Mu2 = [0.6, 0.3]
    # Gaussian covariances
    # direction vectors for constructing the covariance matrix using
    # outer product of a vector with itself then the principal direction
    # of covariance matrix becomes the given vector and its orthogonal
    # complement
    Sigma1_v = [0.3, 0.1]
    Sigma2_v = [0.1, 0.2]

    Sigma1_scale = 5e-1
    Sigma2_scale = 3e-1

    Sigma1_regularization = np.eye(param.nbVarX) * 5e-3
    Sigma2_regularization = np.eye(param.nbVarX) * 1e-2

    # GMM Gaussian Mixture Model
    Mu = np.zeros((param.nbVarX, param.nbGaussian))
    Mu[:, 0] = np.array(Mu1)
    Mu[:, 1] = np.array(Mu2)
    # covariance matrices
    Sigma = np.zeros((param.nbVarX, param.nbVarX, param.nbGaussian))
    # construct the covariance matrix using the outer product
    Sigma[:, :, 0] = (
        np.vstack(Sigma1_v) @ np.vstack(Sigma1_v).T * Sigma1_scale
        + Sigma1_regularization
    )
    Sigma[:, :, 1] = (
        np.vstack(Sigma2_v) @ np.vstack(Sigma2_v).T * Sigma2_scale
        + Sigma2_regularization
    )
    # mixing coefficients priors (summing to one)
    Alpha = np.ones(param.nbGaussian) / param.nbGaussian
    return Mu, Sigma, Alpha


def discrete_gmm(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Discretize given GMM using Fourier basis functions
    rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVarX, param.nbFct, param.nbFct))
    KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
    # Mind the flatten() !!!

    # Explicit description of w_hat by exploiting the Fourier transform
    # properties of Gaussians (optimized version by exploiting symmetries)
    op = hadamard_matrix(2 ** (param.nbVarX - 1))
    op = np.array(op)
    # check the reshaping dimension !!!
    kk = KX.reshape(param.nbVarX, param.nbFct**2) * param.omega

    # Compute fourier basis function weights w_hat for the target distribution given by GMM
    w_hat = np.zeros(param.nbFct**param.nbVarX)
    for j in range(param.nbGaussian):
        for n in range(op.shape[1]):
            MuTmp = np.diag(op[:, n]) @ param.Mu[:, j]
            SigmaTmp = np.diag(op[:, n]) @ param.Sigma[:, :, j] @ np.diag(op[:, n]).T
            cos_term = np.cos(kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
            # Eq.(22) where D=1
            w_hat = w_hat + param.Alpha[j] * cos_term * exp_term
    w_hat = w_hat / (param.L**param.nbVarX) / (op.shape[1])

    # Fourier basis functions (for a discretized map)
    xm1d = np.linspace(param.xlim[0], param.xlim[1], param.nbRes)  # Spatial range
    xm = np.zeros((param.nbGaussian, param.nbRes, param.nbRes))
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
    # Mind the flatten() !!!
    ang1 = (
        KX[0, :, :].flatten().T[:, np.newaxis]
        @ xm[0, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    ang2 = (
        KX[1, :, :].flatten().T[:, np.newaxis]
        @ xm[1, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    phim = np.cos(ang1) * np.cos(ang2) * 2 ** (param.nbVarX)
    # Some weird +1, -1 due to 0 index !!!
    xx, yy = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
    phim = phim * np.tile(HK, (param.nbRes**param.nbVarX, 1)).T

    # Desired spatial distribution
    g = w_hat.T @ phim
    return g

# Define the GP model without derivatives
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)