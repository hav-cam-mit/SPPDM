import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import networkx as nx
from utils import split_data,generate_erdos_renyi_graph,symmetric_fdla_matrix,_gradient,f,grad_error,SPPDM
from utils import generate_ring_graph
import cvxpy as cvx


N = 30 # node number
# dim = 100
# M = 6000 # sample number
# #dim = 100
# size_N =int(M/N) # the number of sample in each agent
# batch_size = 200

num_epochs = 1000
#n = 100  # dimension
rho = 5
sigma = 2
prob =0.3
rng = np.random.RandomState(10)
# X_total, Y_total  = make_regression(n_samples=M, n_features=dim,n_informative=20, random_state=10, noise=sigma**2,
#                       bias=100.0)
dataset = fetch_california_housing()
X_total, Y_total = dataset.data, dataset.target
X_total= StandardScaler().fit_transform(X_total)
M = len(X_total[:,0])
dim = len(X_total[0,:])
size_N =int(M/N) # the number of sample in each agent
batch_size =  688 #688
state_b = np.arange(M)
np.random.shuffle(state_b)
X_total = X_total[state_b]
Y_total = Y_total[state_b]
m = np.ones(N, dtype=int) * size_N
X = split_data(m,X_total)
Y = split_data(m, Y_total)

#G = generate_erdos_renyi_graph(N, prob)
G= generate_ring_graph(N)
x_0 = 1*np.random.rand(dim,N)
x_0_mean = x_0.mean(axis=1)
W, al,A = symmetric_fdla_matrix(G)
A = A.A
AA = A.T.dot(A)
B = np.abs(A)
BB = B.T.dot(B)
D = AA + BB
I = nx.incidence_matrix(G, oriented=True)
degree = nx.degree(G)
SS = -I.todense().T
incedency_matrix = SS.A

ind = nx.adjacency_matrix(G).toarray()
neighbors = [[]]*N
for i in range(N):
    a = []
    for j in range(N):

        if ind[i, j] == 1:
            a.append(j)
    neighbors[i] = a

phi_0 = np.ones((dim, N))
phi_0_inv = np.zeros((N, N))
kappa =4
alpha =0.2

c =1
gamma =40
beta =.1
eta = 0.98
delta=0.1
for i in range(N):
    phi_0[:, i] = phi_0[:, i] / (gamma +  c * D[i, i] + kappa)
    phi_0_inv[i, i] = 1 / (gamma + c * D[i, i] + kappa)

x_SPPDM = x_0.copy()
x_last_SPPDM = x_0.copy()
z_SPPDM = x_0.copy()
pp_SPPDM = np.zeros((dim, N))
gap_SPPDM = np.zeros(num_epochs+1)
objective_SPPDM = np.zeros(num_epochs)

x_SPPDM_mom =x_0.copy()
x_last_SPPDM_mom = x_0.copy()
z_SPPDM_mom = x_0.copy()
pp_SPPDM_mom = np.zeros((dim, N))
gap_SPPDM_mom = np.zeros(num_epochs+1)
objective_SPPDM_mom = np.zeros(num_epochs)





grad_last = np.zeros((dim, N))

W_min_diag = min(np.diag(W))
tmp = (1 - 1e-1) / (1 - W_min_diag)
#_s = W*tmp + np.eye(N) * (1 - tmp)
W_s = (np.eye(N)+W)/2




gap_SPPDM[0] = grad_error(x_SPPDM, pp_SPPDM, X, Y, rho, delta)
gap_SPPDM_mom[0] = grad_error(x_SPPDM_mom, pp_SPPDM_mom, X, Y, rho, delta)


for epoch in range(num_epochs):
    x_SPPDM, pp_SPPDM, z_SPPDM, x_last_SPPDM = SPPDM(x_SPPDM, pp_SPPDM, z_SPPDM, x_last_SPPDM, phi_0, phi_0_inv, X, Y, batch_size, m,
                                                    AA, BB, alpha, gamma, c, kappa, beta, 0, delta,rho)
    x_SPPDM_mom, pp_SPPDM_mom, z_SPPDM_mom, x_last_SPPDM_mom = SPPDM(x_SPPDM_mom, pp_SPPDM_mom, z_SPPDM_mom, x_last_SPPDM_mom, phi_0, phi_0_inv, X, Y,
                                                    batch_size, m,
                                                     AA, BB, alpha, gamma, c, kappa, beta, eta, delta, rho)

    delta_t,delta_t1 = 0.1/(1+epoch),0.01

    gap_SPPDM[epoch+1] = grad_error(x_SPPDM, pp_SPPDM,X,Y,rho,delta)
    gap_SPPDM_mom[epoch+1] = grad_error(x_SPPDM_mom, pp_SPPDM_mom, X, Y, rho, delta)



    objective_SPPDM[epoch]=f(x_SPPDM.mean(axis=1), X_total, Y_total, rho)
    objective_SPPDM_mom[epoch] = f(x_SPPDM_mom.mean(axis=1), X_total, Y_total, rho)

fig1, ax1 = plt.subplots()
plt.grid(True,which="both",ls="--")
plt.semilogy(gap_SPPDM,'gs-',markevery=50,label='Prox-ADMM')
plt.semilogy(gap_SPPDM_mom,'ro-',markevery=50, label='PPDM')

plt.legend(loc='best')
plt.xlabel('Communication round k')
plt.ylabel('Gap')
#plt.savefig("./Figs/gap_agent30_ring_full.eps")
plt.show()

