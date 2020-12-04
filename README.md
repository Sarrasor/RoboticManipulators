# Robotic manipulators
Calculation of Forward Kinematics (FK), Inverse Kinematics (IK), Jacobians, Dynamic Modeling (Euler-Lagrange, Newton-Euler), and trajectory generation (Joint Space polynomial, Joint Space P2P, Cartesian Space Linear) for robotic manipulators.

## Repo contents

* `robots` - Folder with IK and FK solutions. Solution descriptions are in `.md` files
* `utils` - Several useful utils like `SymbolicTransformation` or `TrajectoryGenerator` that can help with matrix multiplication, planning and other Robotics-related stuff
* `tests` - Unit tests
* `Dynamics.md` - Review of dynamic modeling of robotics manipulators
* `TrajectoryPlanning.md` - Review of trajectory planning for robotic manipulators

## How to run

Here are several useful commands to run:

### RR Robot dynamic modeling
`python rr_robot_dynamics.py`

If you want to see how to model RR manipulator with gravity force and make it follow the desired trajectory. Check out `Dynamics.md` to see how it is done.

#### No control

No control signal is applied, just gravity force is acting:

![RR Robot gravity motion](images/rr_robot_gravity.gif)

#### Control

Control signal is applied:

![RR Robot batman trajectory motion](images/rr_robot_trajectory_batman.gif)

### RRR Robot trajectory planning

`python rrr_robot_planning.py`

If you want to see how to perform XYZ Polynomial and Trapezoidal trajectory planning for RRR Robot in Joint and Cartesian Space. Check out `TrajectoryPlanning.md` to see how the planning is done.

#### Polynomial profile

Sample polynomial trajectory: 

![Polynomial profile plots](images/polynomial_profile_plots.png)

Here is how the manipulator will move:

![Polynomial profile motion](images/polynomial_profile_motion.gif)

#### Trapezoidal profile (Joint Space)

Sample trapezoidal trajectory in joint space: 

![Trapezoidal profile plots](images/trapezoidal_profile_plots.png)

Here is how the manipulator will move:

![Trapezoidal profile motion](images/trapezoidal_profile_motion.gif)

#### Trapezoidal profile (Cartesian Space)

Sample trapezoidal trajectory in cartesian space: 

![Trapezoidal cartesian profile plots](images/trapezoidal_cartesian_profile_plots.png)

Their corresponding joint space plots:

![Corresponding joint profile plots](images/trapezoidal_cartesian_joint_profile_plots.png)

Here is how the manipulator will move:

![Trapezoidal cartesian profile motion](images/trapezoidal_cartesian_profile_motion.gif)

### Fanuc Kinematics

`python fanuc_kinematics.py`

If you want to see how to use Fanuc165F forward and inverse kinematics calculation.
Check out `robots/FANUC165F.md` for the solution description:

![Fanuc plot](images/fanuc_plot.png)

### Fanuc Jacobians

`python fanuc_jacobians.py`

If you want to see how to calculate Fanuc165F Jacobian matrix using Scew theory and numerical matrix differentiation methods. The singularity analysis is also presented.
Check out `FanucJacobians.md` for the solution description:

![Fanuc singularity](images/ballerina_singularity.png)

### Jacobian Calculation

`python jacobian_calculation.py`

If you want to see how to use the `JacobianCalculator` class:

![Jacobian calculation](images/jacobian_example.png)

### Symbolic Transformations

`python symbolic_calculation.py`

If you want to see how to use the `SymbolicTransformation` class:

![Symbolic calculation](images/symbolic_calculation.png)

### Unit Tests

`python -m unittest discover`

If you want to run all unit tests:

![Tests](images/tests.png)





