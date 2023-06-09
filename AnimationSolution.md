## Animation by cost minimization 

### Cost Function

The animation is achieved by using the gradient-descent algorithm to solve the following minimization problem: 
$$
\begin{align}
	\hat{\bf \Phi} = \arg_{{{\bf \Phi}}}\min  C\left({\bf \Phi}\right),
	\label{simple_motion_min}
\end{align}
$$
where the cost function $C$ is defined by:
$$
\begin{align}
	C\left({\bf \Phi}\right) = \underbrace{\|{\bf e}\left({\bf \Phi}\right) - {\bf g}\|}_{\text{goal attraction}} + \underbrace{\sum_{i=1}^{n} \mathcal{F}_R\left(\|{\bf e}\left({\bf \Phi}\right) - {\bf o}_i\|\right)}_{\text{obstacle-avoidance penalty}} + \underbrace{\sum_{j=1}^{3} \mathcal{L}\left(\phi_j\right)}_{\text{Joint-range limit}}.
	\label{simple_motion}
\end{align}
$$
Here, ${\bf g} = \left(x_g, y_g, z_g\right)^\mathsf{T}$ is the *goal location*, ${\bf o}_i$ is the location of obstacle $i$. Function ${\bf e}\left({\bf \Phi}\right)$ computes the arm's *forward kinematics* and returns the location of the arm's tip ${\bf e} = \left( e_x, e_y, e_z \right)^\mathsf{T}$, i.e., the *end-effector,* given the arm's joint angles, ${\bf \Phi} = \left( \phi_1, \phi_2,  \phi_3, \phi_4\right)^\mathsf{T}$.   

### Gradient-descent solution for the animation

#### Algorithm 1

1. $\Phi_t \gets (\phi_1, \phi_2, \phi_3, \phi_4)^\mathsf{T}$                                 $\triangleright$ Current joint-angle configuration (i.e., current pose)
2. ${\bf e}_t \gets (e_x, e_y, e_z)^\mathsf{T}$                                      $\triangleright$ Current position of the end-effector
3. ${\bf g} \gets (g_x, g_y, g_z)^\mathsf{T}$                                       $\triangleright$ Location of the goal (or target)
4. $d \gets 0.01$                                                     $\triangleright$ Acceptable distance to the goal
5. $\lambda \gets 0.1$                                                        $\triangleright$ Step of the gradient descent
6. **while**   $\|{\bf e}_t - {\bf g}\| > d$       **do**: 
7. > >  $\Phi_{t+1} \gets \Phi_{t}- \lambda \nabla C(\Phi_t)$                      $\triangleright$ Calculate the new joint-angle configuration 
8. > > `drawRobotArm`($\Phi_{t+1}$)                            $\triangleright$ Draw robot arm at the new configuration 
9. > > ${\bf e}_t \gets {\bf e}\left(\Phi_{t+1}\right)$                                        $\triangleright$ Updated end-effector location  
10. > > $\Phi_{t} \gets \Phi_{t+1}$                                             $\triangleright$ Set current configuration to new configuration 
11.   **end while**

#### Gradient calculation

$$
\nabla C\left(\Phi_t\right) = \frac{\partial C\left(\Phi_t\right)}{\partial \Phi_t} \approx
\begin{bmatrix}
	\frac{\Delta C}{\Delta \phi_1} \\
	\frac{\Delta C}{\Delta \phi_2} \\
	\frac{\Delta C}{\Delta \phi_3} \\
	\frac{\Delta C}{\Delta \phi_4} 
\end{bmatrix} 
= {\scriptsize
\begin{bmatrix}
	\dfrac{{C}\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
              +
               \begin{bmatrix}
               		\Delta \phi_1\\   0\\ 0\\   0
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
                                                              \right)}{\Delta \phi_1} \\\\
	\dfrac{{C}\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0\\   \Delta \phi_2\\ 0\\   0
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
                                                              \right)}{\Delta \phi_2} \\\\
	\dfrac{{C}\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0\\   0\\ \Delta \phi_3\\   0
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
                                                              \right)}{\Delta \phi_3} \\\\
	\dfrac{{C}\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0\\   0\\ 0\\   \Delta \phi_4
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		\phi_1\\   \phi_2\\ \phi_3\\   \phi_4
               \end{bmatrix}
                                                              \right)}{\Delta \phi_4} 
\end{bmatrix}}.
$$

#### Numerical example

Here is a numerical example showing the calculation of the gradient of the cost function for a given join-angle configuration. Assume that the current robot arm joint-angle configuration is $\Phi_t = (10, 5, -15, 20)^\mathsf{T}$ and that we set the incremental variations $\Delta \phi_i = 0.05$. Then, the gradient vector calculation is:  

$$
\nabla C\left(\Phi_t\right) = \frac{\partial C\left(\Phi_t\right)}{\partial \Phi_t} \approx
\begin{bmatrix}
	\frac{\Delta C}{\Delta \phi_1} \\
	\frac{\Delta C}{\Delta \phi_2} \\
	\frac{\Delta C}{\Delta \phi_3} \\
	\frac{\Delta C}{\Delta \phi_4} 
\end{bmatrix} 
= 
{\scriptsize
\begin{bmatrix}
	\dfrac{{C}\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0.05\\   0\\ 0\\   0
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
                                                              \right)}{0.05} \\\\
	\dfrac{{C}\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0\\   0.05 \\ 0\\   0
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
                                                              \right)}{0.05} \\\\
	\dfrac{{C}\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0\\   0\\ 0.05\\   0
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
                                                              \right)}{0.05} \\\\
	\dfrac{{C}\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
              +
               \begin{bmatrix}
               		0\\   0\\ 0\\   0.05
               \end{bmatrix}      
                                                              \right) - C\left(
               \begin{bmatrix}
               		10\\   5\\ -15\\   20
               \end{bmatrix}
                                                              \right)}{0.05} 
\end{bmatrix}.}
$$

Of course, the actual values of the cost, i.e., $C(\cdot)$ depend on forward-kinematics function that returns that location of the end-effector, i.e., ${\bf e}(\Phi)$. The forward-kinematics function is specific for each robot-arm. Also, depending on how the input angles are processed inside the cost function calculation (e.g., rotations that use $sine$ and $cosine$ functions, we will need to convert the values of the angles from degrees to radians before we pass them to the forward-kinematics function. 






##### 