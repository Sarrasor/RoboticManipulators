# Dynamics

We can build dynamical models of robotic manipulators that consider forces and torques acting on the robot, and model gravity and motion of bodies with mass and inertia. Those models can help us to contol manipulators. For example, we can simulate behaviour of our manipulator and find required input torques to perform the desired motion. 

In order to model our robot, we can use one of the two methods:

## Euler-Lagrange

Here is a brief description of the method:

![Euler-Lagrange method](images/euler_lagrange.png)

## Newton-Euler

Here is a brief description of the method:

![Newton-Euler method](images/newton_euler.png)

## RR Robot model

RR Robot Dynamic model:

![RR Robot dynamic model](images/rr_robot_model.png)

## RR Robot model simulation

Since we know the differential equation for the RR manipulator, we can simulate RR Robot now. Here is the result of simulation with zero input torques and forces and gravity force:

![RR Robot gravity motion](images/rr_robot_gravity.gif)

![RR Robot gravity trajectory plot](images/rr_robot_gravity_qs.png)

## Desired trajectory following

Now we want the manipulator to follow the desired trajectory. First of all, we need to generate it. In order to do that, we firstly create the trajectory in the Cartesian space, and then transform it to the Joint space afterwards. Here is an example of the trajectory following task:

![RR Robot ellipse trajectory motion](images/rr_robot_trajectory_ellipse.gif)

![RR Robot ellipse trajectory plot](images/rr_robot_ellipse_qs.png)

![RR Robot ellipse trajectory input](images/rr_robot_ellipse_us.png)

And here is the result of more complicated trajectory following task:

![RR Robot batman trajectory motion](images/rr_robot_trajectory_batman.gif)

![RR Robot batman trajectory plot](images/rr_robot_batman_qs.png)

![RR Robot batman trajectory input](images/rr_robot_batman_us.png)



