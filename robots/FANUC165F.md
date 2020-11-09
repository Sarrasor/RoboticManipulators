# FANUC R-2000iC/165F Manipulator

Here is the description of how to solve Forward and Inverse Kinematics problems for the manipulator.

## Manipulator Description

The task is to calculate Forward and inverse kinematics for **FANUC R-2000iC/165F**

Here is a diagram of the manipulator:

![Fanuc manipulator links](../images/fanuc_links.png)

Here is a scheme with all required dimensions:

![Fanuc manipulator dimensions](../images/fanuc_dimensions.png)

And here is a simplified representation:

![Fanuc manipulator simplified](../images/fanuc_scheme.png)

## Forward kinematics
We are given a joint configuration <img src="https://latex.codecogs.com/gif.latex?%5Cvec%20q%20%3D%20%5Cbegin%7Bbmatrix%7D%20q_1%20%26%20q_2%20%26%20q_3%20%26%20q_4%20%26%20q_5%20%26%20q_6%5Cend%7Bbmatrix%7D%5ET"/> and need to obtain corresponding pose of the end effector  <img src="https://latex.codecogs.com/gif.latex?T_{FK}"/> in homogeneous form.

The simplified representation makes the derivation of forward kinematics prety straightforward. Just follow the links:

![Fanuc Forward kinematics](../images/fanuc_fk.png)


## Inverse kinematics
We are given a pose <img src="https://latex.codecogs.com/gif.latex?T_{IK}"/> and need to find all link configurations <img src="https://latex.codecogs.com/gif.latex?%5Cvec%20q_k"/> that can move effector to pose <img src="https://latex.codecogs.com/gif.latex?T_{IK}"/>

We shall use the Pieper’s method for that:

![Fanuc Pieper's method](../images/fanuc_ik_pieper.png)

Let us consider position first:

![Fanuc Pieper's method](../images/fanuc_ik_t012.png)

Now, for the rotation:

![Fanuc Pieper's method](../images/fanuc_ik_t345.png)
