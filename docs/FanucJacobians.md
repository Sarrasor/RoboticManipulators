# Jacobian Calculation

The task is to calculate the Jacobian matrix for the **FANUC R-2000iC/165F** manipulator

## Robot Description
We use the following representation of the manipulator:

![Fanuc manipulator simplified](../images/fanuc_scheme.png)

## Numeric method

The matrix derivatives solution is described here:

![Numeric Jacobian](../images/numeric_jacobian.png)

And here is a cheatsheet with matrix derivatives, just for reference:

![Matrix Derivatives](../images/matrix_derivatives.png)

## Skew theory method

The skew theory solution is summarized here:

![Scew theory Jacobian](../images/scew_jacobian.png)

## Singularity analysis

There are three known possible singularities:

1) Orientation singularity: <img src="https://latex.codecogs.com/gif.latex?q_4 = 0"/>

![Orientation singularity](../images/orientation_singularity.png)

2) "Ballerina" singularity: Tool pose lies on the <img src="https://latex.codecogs.com/gif.latex?q_0"/> rotation axis

![Ballerina singularity](../images/ballerina_singularity.png)

3) Full extension singularity: <img src="https://latex.codecogs.com/gif.latex?q_4 = 0"/>

![Full extension singularity](../images/full_extension_singularity.png)

And here are possible joint configurations to obtain the singularity cases:

![Singularity configurations](../images/singularity_configurations.png)

