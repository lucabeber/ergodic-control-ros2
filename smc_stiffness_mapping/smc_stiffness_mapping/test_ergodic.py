import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3


import numpy as np
import matplotlib.pyplot as plt

# Append the path of torch and gpytorch to the system path
import sys
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/home/lucabeber/.local/lib/python3.10/site-packages')

import torch
import gpytorch

class StiffnessMappingNode(Node):

    def __init__(self):
        super().__init__('stiffness_mapping_node')
        self.subscription = self.create_subscription(
            Point,
            '/position',
            self.position_callback,
            10)
        self.stiffness_publisher = self.create_publisher(Vector3, '/stiffness', 10)


        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.test_ergodic_callback)

        param = lambda: None # Lazy way to define an empty class in python
        param.nbDataPoints = 2000
        param.min_kernel_val = 1e-8  # upper bound on the minimum value of the kernel
        param.diffusion = 1  # increases global behavior
        param.source_strength = 1  # increases local behavior
        param.obstacle_strength = 0  # increases local behavior
        param.agent_radius = 5  # changes the effect of the agent on the coverage
        param.max_dx = 1.0 # maximum velocity of the agent
        param.max_ddx = 0.1 # maximum acceleration of the agent
        param.dt = 1.0
        param.cooling_radius = (
            1  # changes the effect of the agent on local cooling (collision avoidance)
        )
        param.nbAgents = 1
        param.local_cooling = 0  # for multi agent collision avoidance
        param.dx = 50.0/100.0

        param.nbVarX = 2  # dimension of the space
        param.nbResX = 100 # number of grid cells in x direction
        param.nbResY = 100  # number of grid cells in y direction

        param.nbGaussian = 2

        param.nbFct = 10  # Number of basis functions along x and y
        # Domain limit for each dimension (considered to be 1
        # for each dimension in this implementation)
        param.xlim = [0, 1]
        param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
        param.omega = 2 * np.pi / param.L

        param.nbRes = param.nbResX  # resolution of discretization

        param.alpha = np.array([1, 1]) * param.diffusion

        self.position_x = 37.0
        self.position_y = 5.0

        dim_x_s = param.nbResX * param.dx
        dim_y_s = param.nbResX * param.dx * 22.0 / 50.0

        self.dim_x = param.nbResX * param.dx
        self.dim_y = param.nbResY * param.dx

        L_list = np.array([dim_x_s, dim_y_s])  # boundaries for each dimension

        # Conctruct grid data
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, L_list[0], 26),
            np.linspace(0, L_list[1], 12)
        )

        data = np.genfromtxt('/home/lucabeber/experiment_setup/controller_ws/src/ergodic-control-ros2/smc_stiffness_mapping/ric.csv', delimiter=',', skip_header=0)

        # Save data in np arrays
        # first column
        el_ekf = data[:,0]
        # second column
        visc_elk = data[:,1]
        # third column
        el_ukf = data[:,2]
        # fourth column
        visc_ukf = data[:,3]
        grids_gpr = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx = grids_x[1,1]
        dy = grids_y[1,1]

        # Eliminate the data that differs more than 0.2 one from the previous one
        # EKF 
        for i in range(1, len(el_ekf)):
            if np.abs(el_ekf[i] - el_ekf[i-1]) > 0.2:
                el_ekf[i] = el_ekf[i-1]
                visc_elk[i] = visc_elk[i-1]
                
        # Construct training data
        train_x = torch.tensor(grids_gpr, dtype=torch.float32)
        train_y = torch.tensor(el_ekf, dtype=torch.float32)

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

        self.likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
        self.model_real = GPModel(train_x, train_y, self.likelihood_real)

        # model_real.

        # Training the model
        self.model_real.train()
        self.likelihood_real.train()

        # Initialize the model with the hyperparameters
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(0.004),
            'covar_module.base_kernel.lengthscale': torch.tensor(4.995),
            'covar_module.outputscale': torch.tensor(0.014),
            'mean_module.constant': torch.tensor(0.274)
        }

        self.model_real.initialize(**hypers)

        # Switch to evaluation mode
        self.model_real.eval()
        self.likelihood_real.eval()



    def position_callback(self, msg):
        self.position_x = msg.x
        self.position_y = msg.y

    def test_ergodic_callback(self):
        # Make prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gpr_original_stiff = self.likelihood_real(self.model_real(torch.tensor([[self.position_x, self.position_y]], dtype=torch.float32)))

        stiffness_sample = gpr_original_stiff.mean.cpu()

        # Publish stiffness
        tmp = stiffness_sample.numpy()[0]+1e-8
        msg = Vector3()
        msg.x = self.position_x
        msg.y = self.position_y
        msg.z = tmp
        self.stiffness_publisher.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = StiffnessMappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()