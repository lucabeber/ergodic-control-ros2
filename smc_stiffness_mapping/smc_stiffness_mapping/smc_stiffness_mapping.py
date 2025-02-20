#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
import sys, os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'smc_stiffness_mapping'))
from smc_lib import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import time

class ErgodicExploration(Node):

    def __init__(self):
        super().__init__('ergodic_exploration')

        self.subscription_stiff = self.create_subscription(
            Vector3,
            '/stiffness',
            self.stiffness_callback,
            10
        )

        # Create publisher for the position to be sent to the robot
        self.publisher_position = self.create_publisher(Point, '/position', 10)

        self.first_stiff = False

        self.current_x = 0.0
        self.current_y = 0.0

        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.ergodic_exploration_callback)

        # Parameters
        # ===============================
        self.param = lambda: None # Lazy way to define an empty class in python
        self.param.nbDataPoints = 6000
        self.param.min_kernel_val = 1e-8  # upper bound on the minimum value of the kernel
        self.param.diffusion = 1  # increases global behavior
        self.param.source_strength = 1 # increases local behavior
        self.param.obstacle_strength = 0  # increases local behavior
        self.param.agent_radius = 5  # changes the effect of the agent on the coverage
        self.param.max_dx = 0.1 # maximum velocity of the agent
        self.param.max_ddx = 0.001 # maximum acceleration of the agent
        self.param.cooling_radius = (
            1  # changes the effect of the agent on local cooling (collision avoidance)
        )
        self.param.nbAgents = 1
        self.param.local_cooling = 0  # for multi agent collision avoidance
        self.param.nbResX = 100 # number of grid cells in x direction
        self.param.nbResY = 100  # number of grid cells in y direction
        self.param.dx = 50.0/self.param.nbResX
        self.param.dy = 50.0/self.param.nbResY

        self.param.nbVarX = 2  # dimension of the space
        self.param.nbResX = 100 # number of grid cells in x direction
        self.param.nbResY = 100  # number of grid cells in y direction

        self.param.dt = 1.0  # time step
        self.param.nbFct = 10  # Number of basis functions along x and y
        # Domain limit for each dimension (considered to be 1
        # for each dimension in this implementation)
        self.param.xlim = [0, 1]
        self.param.L = (self.param.xlim[1] - self.param.xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
        self.param.omega = 2 * np.pi / self.param.L

        self.param.nbRes = self.param.nbResX  # resolution of discretization

        self.param.alpha = np.array([1, 1]) * self.param.diffusion

        self.G = np.zeros((self.param.nbResX, self.param.nbResY))

        self.dim_x = self.param.nbResX * self.param.dx
        self.dim_y = self.param.nbResY * self.param.dx

        gradient_times_variance = np.ones(self.param.nbResX * self.param.nbResY)
        gradient_times_variance /= ( sum(gradient_times_variance) * self.param.dx * self.param.dy)


        # self.param.Mu, self.param.Sigma, self.param.Alpha = get_GMM(self.param)
        g = gradient_times_variance
        self.G = np.reshape(g, [self.param.nbResX, self.param.nbResY])
        # Reverse the array to have the same orientation as the grid
        self.G = np.abs(self.G)  # there is no negative heat


        # Initialize heat equation related fields
        # ===============================
        # precompute everything we can before entering the loop
        self.coverage_arr = np.zeros((self.param.nbResX, self.param.nbResY, self.param.nbDataPoints))
        self.heat_arr = np.zeros((self.param.nbResX, self.param.nbResY, self.param.nbDataPoints))
        self.local_arr = np.zeros((self.param.nbResX, self.param.nbResY, self.param.nbDataPoints))
        self.goal_density_arr = np.zeros((self.param.nbResX, self.param.nbResY, self.param.nbDataPoints))

        self.param.height, self.param.width = self.G.shape

        self.param.area = self.param.dx * self.param.width * self.param.dx * self.param.height

        self.goal_density = normalize_mat(self.G)

        self.coverage_density = np.zeros((self.param.height, self.param.width))
        self.heat = np.array(self.goal_density)

        max_diffusion = np.max(self.param.alpha)
        self.param.dt = min(
            1.0, (self.param.dx * self.param.dx) / (4.0 * max_diffusion)
        )  # for the stability of implicit integration of Heat Equation
        self.coverage_block = agent_block(self.param.min_kernel_val, self.param.agent_radius)
        self.cooling_block = agent_block(self.param.min_kernel_val, self.param.cooling_radius)
        self.param.kernel_size = self.coverage_block.shape[0]

        self.agent_arr = np.zeros((self.param.nbAgents, self.param.nbDataPoints, 2))
        # HEDAC Loop
        # ===============================
        # do absolute minimum inside the loop for speed
        self.fig, self.ax = plt.subplots(1, 3, figsize=(16, 8))
        self.frames = []
        # Set up the video writer
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('animated_plot_video.mp4', self.fourcc, 10, (2048, 1080))  # Change size to 2k resolution


        self.t = 0

        # Conctruct grid data
        self.grids_x, self.grids_y = np.meshgrid(
            np.linspace(0, self.dim_x, 100),
            np.linspace(0, self.dim_y, 100)
        )

        self.grids = np.array([self.grids_x.ravel(), self.grids_y.ravel()]).T

    
    def stiffness_callback(self, msg):
        # self.get_logger().info('Received stiffness: %s' % msg)
        self.current_x = msg.x * 1000.0
        self.current_y = msg.y * 1000.0
        self.current_stiff = msg.z

        if self.first_stiff == False:
            self.init_guassian_process()
            self.first_stiff = True

    def ergodic_exploration_callback(self):

        if self.first_stiff == False:
            return
        
        # cooling of all the agents for a single timestep
        # this is used for collision avoidance bw/ agents
        local_cooling = np.zeros((self.param.height, self.param.width))
        for agent in self.agents:
            # find agent pos on the grid as integer indices
            p = agent.x
            adjusted_position = p / self.param.dx
            col, row = adjusted_position.astype(int)

            # each agent has a kernel around it,
            # clamp the kernel by the grid boundaries
            row_indices, row_start_kernel, num_kernel_rows = clamp_kernel_1d(
            row, 0, self.param.height, self.param.kernel_size
            )
            col_indices, col_start_kernel, num_kernel_cols = clamp_kernel_1d(
            col, 0, self.param.width, self.param.kernel_size
            )

            # add the kernel to the coverage density
            # effect of the agent on the coverage density
            self.coverage_density[row_indices, col_indices] += self.coverage_block[
            row_start_kernel : row_start_kernel + num_kernel_rows,
            col_start_kernel : col_start_kernel + num_kernel_cols,
            ]

            # local cooling is used for collision avoidance between the agents
            # so it can be disabled for speed if not required
            # if self.param.local_cooling != 0:
            #     local_cooling[row_indices, col_indices] += self.cooling_block[
            #         row_start_kernel : row_start_kernel + num_kernel_rows,
            #         col_start_kernel : col_start_kernel + num_kernel_cols,
            #     ]
            # local_cooling = normalize_mat(local_cooling)

        self.coverage = normalize_mat(self.coverage_density)


        # this is the part we introduce exploration problem to the Heat Equation
        diff = self.goal_density - self.coverage
        sign = np.sign(diff)
        source = np.maximum(diff, 0) ** 2
        source = normalize_mat(source) * self.param.area

        current_heat = np.zeros((self.param.height, self.param.width))

        # 2-D heat equation (Partial Differential Equation)
        # In 2-D we perform this second-order central for x and y.
        # Note that, delta_x = delta_y = h since we have a uniform grid.
        # Accordingly we have -4.0 of the center element.

        # At boundary we have Neumann boundary conditions which assumes
        # that the derivative is zero at the boundary. This is equivalent
        # to having a zero flux boundary condition or perfect insulation.
        current_heat[1:-1, 1:-1] = self.param.dt * (
            (
            +self.param.alpha[0] * offset(self.heat, 1, 0)
            + self.param.alpha[0] * offset(self.heat, -1, 0)
            + self.param.alpha[1] * offset(self.heat, 0, 1)
            + self.param.alpha[1] * offset(self.heat, 0, -1)
            - 4.0 * offset(self.heat, 0, 0)
            )
            / (self.param.dx * self.param.dx)
            + self.param.source_strength * offset(source, 0, 0)
            - self.param.local_cooling * offset(local_cooling, 0, 0)
        ) + offset(self.heat, 0, 0)

        self.heat = current_heat.astype(np.float32)

        # Calculate the first derivatives mind the order x and y
        gradient_y, gradient_x = np.gradient(self.heat, 1, 1)

        for agent in self.agents:
            grad = calculate_gradient(
            agent,
            gradient_x,
            gradient_y,
            self.param
            )
            local_heat = bilinear_interpolation(current_heat, agent.x)
            agent.update(grad)

        self.coverage_arr[..., self.t] = self.coverage
        self.heat_arr[..., self.t] = self.heat
        self.goal_density_arr[..., self.t] = self.goal_density
        # Save agent positions
        for i, agent in enumerate(self.agents):
            self.agent_arr[i, self.t] = agent.x

            # Send the position to the robot
            msg = Point()
            msg.x = agent.x[0]
            msg.y = agent.x[1]
            self.publisher_position.publish(msg)

        # update the pdf every 10 steps
        if self.t % 50 == 0:
            start_time = self.get_clock().now()
            # Extract the trajectory
            self.sample_points = torch.cat([self.sample_points, torch.tensor([self.current_x, self.current_y], dtype=torch.float32).reshape(1, -1)], dim=0)
            self.stiffness_points = torch.cat((self.stiffness_points, torch.tensor([self.current_stiff], dtype=torch.float32)))
            # self.get_logger().info('Extract trajectory: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            # Construct training data
            self.train_x = self.sample_points.clone()
            self.train_y = self.stiffness_points.clone()
            # self.get_logger().info('Construct training data: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            # Training the model
            self.model.train()
            self.likelihood.train()
            self.model.set_train_data(self.train_x, self.train_y, strict=False)
            # self.get_logger().info('Train model: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            # Switch to evaluation mode
            self.model.eval()
            self.likelihood.eval()
            # self.get_logger().info('Switch to evaluation mode: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            L_list = np.array([self.dim_x, self.dim_y])
            test_x1 = torch.linspace(0.0, L_list[0], 100)
            test_x2 = torch.linspace(0.0, L_list[1], 100)
            test_x1, test_x2 = torch.meshgrid(test_x1, test_x2)
            self.test_x = torch.cat([test_x2.reshape(-1, 1), test_x1.reshape(-1, 1)], dim=1)
            # self.get_logger().info('Prepare test data: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.observed_pred = self.likelihood(self.model(self.test_x))
            # self.get_logger().info('Make predictions: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            self.test_x.requires_grad_(True)
            pred = self.model(self.test_x)
            mean = pred.mean
            mean.backward(torch.ones(mean.shape))
            var = pred.variance
            # self.get_logger().info('Compute gradients: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            gradient_magnitude = torch.sqrt(self.test_x.grad[:, 0]**2 + self.test_x.grad[:, 1]**2)
            magnitude = gradient_magnitude.detach().numpy() / sum(gradient_magnitude.detach().numpy())
            mean = self.observed_pred.mean.cpu().numpy() / sum(self.observed_pred.mean.cpu().numpy())
            varo = var.detach().numpy()
            # self.get_logger().info('Compute magnitude: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            border_threshold = 0.05
            border_mask_x = np.minimum(self.test_x[:, 0].cpu().detach().numpy() / (border_threshold * self.dim_x),
                                       (1 - self.test_x[:, 0].cpu().detach().numpy() / self.dim_x) / border_threshold)
            border_mask_y = np.minimum(self.test_x[:, 1].cpu().detach().numpy() / (border_threshold * self.dim_y),
                                       (1 - self.test_x[:, 1].cpu().detach().numpy() / self.dim_y) / border_threshold)
            border_mask = np.minimum(border_mask_x, border_mask_y)
            border_mask = np.clip(border_mask, 0, 1)
            varo *= border_mask
            # self.get_logger().info('Apply border mask: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            tmp = (gradient_magnitude.detach().numpy()) - 1.5 * np.mean(gradient_magnitude.detach().numpy())
            gradient_times_variance = np.maximum(tmp * 1.5, np.zeros(100 * 100)) + varo
            gradient_times_variance /= (sum(gradient_times_variance) * self.param.dx * self.param.dy)
            # self.get_logger().info('Compute gradient times variance: %f' % (self.get_clock().now() - start_time).nanoseconds)

            # start_time = self.get_clock().now()
            g = gradient_times_variance
            self.G = np.reshape(g, [self.param.nbResX, self.param.nbResY])
            self.G = np.abs(self.G)
            self.goal_density = normalize_mat(self.G)
            # self.get_logger().info('Update target distribution: %f' % (self.get_clock().now() - start_time).nanoseconds)
            

            self.ax[0].cla()
            self.ax[0].contourf(self.grids_x, self.grids_y, self.goal_density, cmap="gray_r")
            # Print trajectory
            for i, agent in enumerate(self.agents):
                self.ax[0].plot(self.agent_arr[i, :self.t, 0], self.agent_arr[i, :self.t, 1], color="black")
            self.ax[0].set_aspect("equal", "box")
            self.ax[0].set_xlabel('x (mm)')
            self.ax[0].set_ylabel('y (mm)')
            self.ax[0].set_title('Ergodic trajectory and EID')


            # # w_hat
            self.ax[1].cla()
            c = self.ax[1].tricontourf(self.test_x[:, 0].cpu().detach().numpy(), self.test_x[:, 1].cpu().detach().numpy(), self.observed_pred.mean.cpu().numpy(), cmap='viridis')
            # fig.colorbar(c, ax=ax[1], label='Elastic modulus (GPa)')
            self.ax[1].set_aspect('equal', adjustable='box')
            self.ax[1].set_xlabel('x (mm)')
            self.ax[1].set_ylabel('y (mm)')
            self.ax[1].set_title('EKF Elastic modulus prediction')
            # ax[1].set_xticks([])
            # ax[1].set_yticks([])

            # # w
            self.ax[2].cla()
            c = self.ax[2].tricontourf(self.test_x[:, 0].cpu().detach().numpy(), self.test_x[:, 1].cpu().detach().numpy(), varo, cmap='viridis')
            # fig.colorbar(c, ax=ax, label='Elastic modulus (GPa)')
            self.ax[2].set_aspect('equal', adjustable='box')
            self.ax[2].set_xlabel('x (mm)')
            self.ax[2].set_ylabel('y (mm)')
            self.ax[2].set_title('EKF Elastic modulus covariance')

            plt.pause(0.001)
            # Convert the figure to an OpenCV-compatible image and store in frames list
            frame = fig_to_cv2(self.t, self.fig)
            self.frames.append(frame)


        self.t += 1

        # Print the current time step
        # get current ros time
        now = self.get_clock().now()
        # get the time difference
        delta = now - self.prev_time
        # print the time difference
        self.get_logger().info('Time step: %f' % (delta.nanoseconds/1e9))
        # update the previous time
        self.prev_time = now

        if self.t == self.param.nbDataPoints:
            self.get_logger().info('Exploration finished')
            self.timer.cancel()
            plt.close()
            self.print_plots()

        
    def init_guassian_process(self):
        self.prev_time = self.get_clock().now()

        x0 = np.array([self.current_x, self.current_y])
        train_x1 = torch.tensor(x0[0], dtype=torch.float32)
        train_x2 = torch.tensor(x0[1], dtype=torch.float32)
        train_x1, train_x2 = torch.meshgrid(train_x1, train_x2)
        train_x = torch.cat([train_x1.reshape(-1, 1), train_x2.reshape(-1, 1)], dim=1)

        train_y = torch.tensor(self.current_stiff, dtype=torch.float32)


        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(train_x, train_y, self.likelihood)

        # Training the model
        self.model.train()
        self.likelihood.train() 



        # Initialize the model with the hyperparameters
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(0.004),
            'covar_module.base_kernel.lengthscale': torch.tensor(4.995),
            'covar_module.outputscale': torch.tensor(0.014),
            'mean_module.constant': torch.tensor(0.274)
        }

        self.model.initialize(**hypers)

        self.agents = []
        agent = SecondOrderAgent(x=x0, nbDataPoints=self.param.nbDataPoints,max_dx=self.param.max_dx,max_ddx=self.param.max_ddx)
        # agent = FirstOrderAgent(x=x, dim_t=cfg.timesteps)
        rgb = np.random.uniform(0, 1, 3)
        agent.color = np.concatenate((rgb, [1.0]))  # append alpha value
        self.agents.append(agent)

        self.sample_points = torch.tensor(agent.x, dtype=torch.float32).reshape(1, -1)

        self.stiffness_points = torch.tensor([self.current_stiff], dtype=torch.float32)

        self.get_logger().info('Initialization finished')

    def print_plots(self):
        # Create a video of the plots
        # ===============================
        for frame in self.frames:
            # Resize the frame to match the video size
            frame_resized = cv2.resize(frame, (2048, 1080))
            # Write the frame to the video
            self.out.write(frame_resized)

        # Release the video writer and clean up
        self.out.release()
        cv2.destroyAllWindows()

        # # Plot
        # # ===============================
        # fig, self.ax = plt.subplots(1, 3, figsize=(16, 8))

        # self.ax[0].set_title("Agent trajectory and desired GMM")
        # # Required for plotting discretized GMM
        # xlim_min = 0
        # xlim_max = self.param.nbResX * self.param.dx
        # xm1d = np.linspace(xlim_min, xlim_max, self.param.nbResX)  # Spatial range
        # xm = np.zeros((2, self.param.nbResX, self.param.nbResY))
        # xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
        # X = np.squeeze(xm[0, :, :])
        # Y = np.squeeze(xm[1, :, :])

        # self.ax[0].contourf(X, Y, self.G, cmap="gray_r") # plot discrete GMM
        # # Plot agent trajectories
        # for agent in self.agents:
        #     self.ax[0].plot(
        #     agent.x_arr[0, 0], agent.x_arr[0, 1], marker=".", color="black", markersize=10
        #     )
        #     self.ax[0].plot(
        #     agent.x_arr[:, 0],
        #     agent.x_arr[:, 1],
        #     color="black",
        #     linewidth=1,
        #     )
        # self.ax[0].set_aspect("equal", "box")

        # self.ax[1].set_title("Exploration goal (heat source), explored regions at time t")
        # arr = self.goal_density - self.coverage_arr[..., -1]
        # arr_pos = np.where(arr > 0, arr, 0) 
        # arr_neg = np.where(arr < 0, -arr, 0)
        # self.ax[1].contourf(X, Y, arr_pos, cmap='gray_r')
        # # Plot agent trajectories
        # for agent in self.agents:
        #     self.ax[1].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linewidth=10, color="blue", label="agent footprint") # sensor footprint
        #     self.ax[1].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linestyle="--", color="black", label='agent path') # trajectory line
        # self.ax[1].legend(loc="upper left")
        # self.ax[1].set_aspect("equal", "box")

        # self.ax[2].set_title("Gradient of the potential field")
        # gradient_y, gradient_x = np.gradient(self.heat_arr[..., -1])
        # self.ax[2].quiver(X, Y, gradient_x, gradient_y, scale=15, units='xy') # Scales the length of the arrow inversely
        # # self.ax[2].quiver(X, Y, gradient_x, gradient_y)

        # # Plot agent trajectories
        # for agent in self.agents:
        #     self.ax[2].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linestyle="--", color="black") # trajectory line
        #     self.ax[2].plot(
        #     agent.x_arr[0, 0], agent.x_arr[0, 1], marker=".", color="black", markersize=10
        #     )
        # self.ax[2].set_aspect("equal", "box")

        # plt.show()

        # # Plot how the gradient field change over time using heat_arr and coverage_arr
        # # ===============================
        # fig, self.ax = plt.subplots(1, 2, figsize=(16, 8))

        # self.ax[0].set_title("Gradient of the potential field over time")
        # self.ax[1].set_title("Exploration goal (heat source), explored regions over time")
        # self.ax[0].set_aspect('equal', adjustable='box')
        # self.ax[1].set_aspect('equal', adjustable='box')
        # for t in range(self.param.nbDataPoints):
        #     gradient_y, gradient_x = np.gradient(self.heat_arr[..., t])
        #     self.ax[0].quiver(X, Y, gradient_x, gradient_y)
        #     arr = self.goal_density_arr[..., t] - self.coverage_arr[..., t]
        #     arr_pos = np.where(arr > 0, arr, 0)
        #     arr_neg = np.where(arr < 0, -arr, 0)
        #     self.ax[1].contourf(X, Y, arr_pos, cmap='gray_r')
        #     for agent in self.agents:
        #         self.ax[1].plot(agent.x_arr[:t, 0], agent.x_arr[:t, 1], linestyle="--", color="black")
        #         self.ax[1].plot(agent.x_arr[t, 0], agent.x_arr[t, 1], marker=".", color="black", markersize=10)
        #     plt.pause(0.01)
        #     self.ax[0].clear()
        #     self.ax[1].clear()



def main(args=None):
    rclpy.init(args=args)

    ergodic_exploration = ErgodicExploration()

    rclpy.spin(ergodic_exploration)

    ergodic_exploration.destroy_node()
    rclpy.shutdown()
    
