#from cv2 import compare
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
#from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
from datetime import datetime
#from csv import writer
#from pymoo.factory import get_selection
#from pymoo.core.selection import Selection


start = datetime.now() # 

# Import mean return vector and covariance matrix

mreturn = np.loadtxt("mean_return84.csv", delimiter = ",")
cvm = np.loadtxt("covariance_matrix84.csv", delimiter = ",")


cardinality_constraint = 2
cardinality_threshold = 1e-10 # Maximum allocation to assets not included within cardinality constraint
asset_sum_threshold = 1e-5

# NSGA-II algorithm parameters:
population_size = 50
number_of_generations = 150


# Initial sample population to assist convergence to cardinality constraint

init_pop = np.zeros((population_size,len(mreturn)))

for i in range(population_size):
    init_pop[i,:cardinality_constraint] = np.random.rand(cardinality_constraint)
    init_pop[i]=init_pop[i]/np.sum(init_pop[i]) # Normalise
    np.random.shuffle(init_pop[i]) # Shuffle


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("CV").min())


class MyProblem(ElementwiseProblem):

    def __init__(self, mreturn, cvm):
        super().__init__(n_var = len(mreturn),
                         n_obj = 2,
                         n_constr = 2,
                         xl = np.array([0 for asset in mreturn]),
                         xu = np.array([1 for asset in mreturn]))

        self.mreturn = mreturn
        self.cvm = cvm
        self.K = cardinality_constraint

    def _evaluate(self, x, out, *args, **kwargs):

        
        f1 = np.dot(np.array(x).T, (np.dot(self.cvm, np.array(x)))) # Risk/variance objective function
        
        f2 = -(np.dot(np.array(x), self.mreturn)) # Mean return objective function

        g1 = sum(asset > cardinality_threshold for asset in x) - self.K # Cardinality constraint function

        g2 = ((sum(x) - 1) ** 2) - asset_sum_threshold
        
        
        out["F"] = np.column_stack([f1, f2])
        out["G"] = [g1, g2]


#selection = get_selection('random')

problem = MyProblem(mreturn, cvm)

algorithm = NSGA2(pop_size = population_size, sampling = init_pop)

res = minimize(problem,
               algorithm,
               ("n_gen", number_of_generations),
               callback = MyCallback(),
               verbose = True,
               save_history = False,
               seed = None)



for count in range(population_size):
    print(f"X: {res.X[count]}, sum: {res.X[count].sum()}")
    print(f"F: {res.F[count]}")

print(f"Algorithm run time: {datetime.now() - start}")

# General Matplotlib parameters for plot appearance
#plt.rcParams["font.family"] = "serif"
#mpl.rcParams['font.sans-serif'] = 'SimHei'

plt.rc('font', size = 14)
plt.rc('axes', titlesize = 13)
plt.rc('axes', labelsize = 25)
#plt.rc('axes', facecolor = 'gainsboro')
plt.rc('xtick', labelsize = 11)
plt.rc('ytick', labelsize = 11)
#plt.style.use('tableau-colorblind10')


# Plot of non-dominated solutions for objective functions

plt.figure(figsize = (13, 10))# facecolor = 'azure'
plt.scatter((res.F[:, 0]), (-res.F[:, 1]), s = 40, marker = "o")# facecolors = 'black', edgecolors = 'silver'
plt.title(f"42 Asset Long/Short Portfolio with Initial Sampling & Tournament Selection \n Population = {res.algorithm.pop_size}, Generations = {res.algorithm.n_gen}, No Normalisation \n Constraints: Cardinality (K = {cardinality_constraint}, Threshold = {cardinality_threshold}), Weighting Sum = 1 (Threshold {asset_sum_threshold})", pad = 5)
plt.xlabel("Portfolio Risk (Variance of Return)", labelpad = 20)
plt.ylabel("Expected Return (Daily)", labelpad = 20)
plt.grid()
#plt.savefig(f"K={cardinality_constraint}_pop={population_size}_ngen={number_of_generations}_{start}.png")
plt.show()



