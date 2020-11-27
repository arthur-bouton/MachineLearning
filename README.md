# Various machine learning scripts and tools

![](cartpole_ppo_1.gif?raw=true "Starting from the same initial position it has been trained with") | ![](cartpole_ppo_2.gif?raw=true "Starting from a different position than the training one")
:-:|:-:

<p align="center"><i>Policy obtained by <a href="ppo_cartpole.py">ppo_cartpole.py</a> with the Proximal Policy Optimization algorithm implemented in <a href="PPO.py">PPO.py</a>.</i></p>


## Files

- **[looptools.py](looptools.py)**: Tools to monitor, extract data and have control during algorithm progress loops. It contains:
	- A context manager which allows the SIGINT signal to be processed asynchronously.
	- A container-like class to plot variables incrementally on a persistent figure.
	- An iterable class to extract numerical data from a file.
- **[neural_networks.py](neural_networks.py)**: Implementation of scalable Multilayer Perceptron (MLP) and Radial Basis Function network (RBF) with numpy only.
- **[sumtree_sampler.py](sumtree_sampler.py)**: A sum tree structure to efficiently sample items according to their relative priorities.
- **[pendulum.py](pendulum.py)**: A simple pendulum to be controlled by a torque at its hinge.
- **[cartpole.py](cartpole.py)**: A free pendulum mounted on a cart which can be controlled either via its lateral speed by means of an embedded feedback controller or by the lateral force applied to it.
- **[CACLA_pendulum.py](CACLA_pendulum.py)**: Implementation of the Continuous Actor Critic Learning Automaton (CACLA) [1] to swing up a pendulum using only numpy.
- **Deep Deterministic Policy Gradient:**
	- **[DDPG_vanilla.py](DDPG_vanilla.py)**: Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm [2] with TensorFlow 1.
	- **[DDPG_PER.py](DDPG_PER.py)**: Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm [2] with TensorFlow 1 and enhanced with Prioritized Experience Replay (PER) [3].
	- **[ddpg_pendulum.py](ddpg_pendulum.py)**: Training example of the DDPG algorithm to swing up the pendulum.
	- **[ddpg_cartpole.py](ddpg_cartpole.py)**: Training example of the DDPG algorithm to swing up the cart-pole.
- **Proximal Policy Optimization:**
	- **[PPO.py](PPO.py)**: Multithreaded implementation of the Proximal Policy Optimization (PPO) algorithm [4] with TensorFlow 1.
	- **[ppo_pendulum.py](ppo_pendulum.py)**: Training example of the PPO algorithm to swing up the pendulum using multithreaded workers.
	- **[ppo_cartpole.py](ppo_cartpole.py)**: Training example of the PPO algorithm to swing up the cart-pole using workers running in the main thread.
- **Soft Actor-Critic:**
	- **[SAC.py](SAC.py)**: Implementation of the Soft Actor-Critic (SAC) algorithm with automated entropy temperature adjustment [5] with TensorFlow 2.
	- **[sac_pendulum.py](sac_pendulum.py)**: Training example of the SAC algorithm to swing up the pendulum.
	- **[sac_cartpole.py](sac_cartpole.py)**: Training example of the SAC algorithm to swing up the cart-pole.
- **[tf_cpp_binding](tf_cpp_binding)**: Contains a C++ template class to bind TensorFlow native C API in order to easily import and use trained models.
- **[LQR.py](LQR.py)**: Linear-Quadratic Regulators for finite or infinite horizons and continuous or discrete times.
- **[lm_slsqp_cartpole.py](lm_slsqp_cartpole.py)**: Automated control synthesis to swing up a cart-pole for which the physical parameters are unknown. The parameter identification is performed by a non-linear regression, the trajectory planning is based on a direct collocation method using non-linear programming and the trajectory tracking is ensured by LQR control.
- **[quadratures.py](quadratures.py)**: Contains a class providing Gauss-Lobatto quadratures and barycentric Lagrange interpolation.

[1] Van Hasselt, Hado, and Marco A. Wiering. "Reinforcement learning in continuous action spaces."<br />
    2007 IEEE International Symposium on Approximate Dynamic Programming and Reinforcement Learning. IEEE, 2007.<br />
[2] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).<br />
[3] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).<br />
[4] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).<br />
[5] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).


### Dependencies

All you will need are the following packages:

`$ pip install scipy matplotlib tqdm tensorflow tensorflow_probability`

<br />
<br />


## Examples of use of the looptools module


### Installation

To install the module for the current user, run in a terminal:

`$ pip install . --user`


### Monitor and stop a progress loop

For example, to monitor the progress of a running algorithm by plotting the evolution of two variables *reward* and *loss*, using a logarithmic scale for the second one, you can do:

	from looptools import Loop_handler, Monitor

	monitor = Monitor( [ 1, 1 ], titles=[ 'Reward', 'Loss' ], log=2 )

	with Loop_handler() as interruption :
		for i in range( 1000 ) :

			(long computation of the next reward and loss)

			monitor.add_data( reward, loss )

			if interruption() :
				break
	
	(clean completion of the algorithm, backup of the data...)

The `Loop_handler` context manager allows you to stop the iterations with Ctrl+C in a nice way so that the script can carry on after the loop.

The `Monitor` works as a container, so it accepts indexing in order to:
- Modify plotted values:<br />
`monitor[:100] = 0` (set the first 100 data points to a same value)<br />
`monitor[:100] = range( 100 )` (set the first 100 values with an iterable)
- Remove data points from the graph:<br />
`del monitor[:100]` (remove the 100 first values)
- Crop the plotted data:<br />
`monitor( *monitor[100:] )` (keep only the 100 latest data points)<br />
Which is equivalent to:<br />
`data = monitor[100:]`<br />
`monitor.clear()`<br />
`monitor.add_data( *data )`


### Read and plot data from a file

Let's say that you have a file *data.txt* storing the data like this:

	Data recorded at 20:57:08 GMT the 25 Aug. 91
	time: 0.05 alpha: +0.54 beta: +0.84 gamma: +1.55
	time: 0.10 alpha: -0.41 beta: +0.90 gamma: -2.18
	time: 0.15 alpha: -0.98 beta: +0.14 gamma: -0.14
	time: 0.20 alpha: -0.65 beta: -0.75 gamma: +1.15
	...

To extract the time and the variables alpha, beta and gamma, you can either:
- Let `Datafile` identify the columns with numerical values while filtering if necessary the relevant lines with a regex or the number of expected columns:<br />
`datafile = Datafile( 'data.txt', filter='^time:' )` or<br />
`datafile = Datafile( 'data.txt', ncols=8 )`
- Specify the columns where to look for the data with a list:<br />
`datafile = Datafile( 'data.txt', [ 2, 4, 6, 8 ] )`
- Specify the columns with an iterable or a list of iterables:<br />
`datafile = Datafile( 'data.txt', range( 2, 9, 2 ) )`
- Specify the columns with a string that will be processed by the function `strange()` provided by this module as well:<br />
`datafile = Datafile( 'data.txt', '2:8:2' )`

If the file stores the data in CSV, you have to specify that the column separator is a comma with the argument `sep=','`.

Therefore, to plot the data straight from the file, you can do:

	from looptools import Datafile, Monitor

	datafile = Datafile( 'data.txt', [ 2, 4, 6, 8 ] )
	all_the_data = datafile.get_data()

	# Plot for example the two variables alpha and beta on a same graph and
	# the third variable gamma on a second graph below using a dashed line:
	monitor = Monitor( [ 2, 1 ],
	                   titles=[ 'First graph', 'Second graph' ],
	                   labels=[ '$\\alpha$', '$\\beta$', '$\gamma$' ],
	                   plot_kwargs={3: {'ls':'--'}} )
	monitor.add_data( *all_the_data )

Or if you want to iterate over the rows:

	for time, alpha, beta, gamma in datafile :
		monitor.add_data( time, alpha, beta, gamma )

If you want to create a pandas DataFrame from these data, you may need to transpose the representation of rows and columns by using the method `get_data_by_rows`:

	import pandas as pd

	df = pd.DataFrame( datafile.get_data_by_rows(), columns=[ 'time', 'alpha', 'beta', 'gamma' ] )


For further information, please refer to the docstrings in [looptools.py](looptools.py).
