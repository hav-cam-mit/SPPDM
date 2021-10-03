import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import networkx as nx

import cvxpy as cvx

def split_data( m, X):
    '''Helper function to split data according to the number of training samples per agent.'''
    cumsum = m.cumsum().astype(int).tolist()
    inds = zip([0] + cumsum[:-1], cumsum)
    return [X[start:end] for (start, end) in inds]  # Return the reference of data, which is contiguous
def generate_ring_graph(n_agent):
    '''Generate ring connectivity graph.'''

    G = nx.cycle_graph(n_agent)

    # Update number of edges of the actual graph
    return G

def generate_erdos_renyi_graph(n_agent, prob):
    '''Generate connected connectivity graph according to the params.'''

    if prob < 2 / (n_agent - 1):
        print("Need higher probability to create a connected graph!")
        exit(-1)

    G = nx.erdos_renyi_graph(n_agent, prob)
    if nx.is_connected(G):
        # Update number of edges of the actual graph
        return G
    else:
        generate_erdos_renyi_graph(n_agent,prob)
def symmetric_fdla_matrix(G):
    n = G.number_of_nodes()
    I = nx.incidence_matrix(G,oriented=True)
    A = -I.todense().T

    ind = nx.adjacency_matrix(G).toarray() + np.eye(n)
    ind = ~ind.astype(bool)

    average_matrix = np.ones((n, n)) / n
    one_vec = np.ones(n)

    W = cvx.Variable((n, n))

    if ind.sum() == 0:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W == W.T,
                                cvx.sum(W, axis=1) == one_vec
                            ])
    else:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W[ind] == 0,
                                W == W.T,
                                cvx.sum(W, axis=1) == one_vec
                            ])
    prob.solve()

    W = W.value
    W = (W + W.T) / 2
    W[ind] = 0
    W -= np.diag(W.sum(axis=1) - 1)
    alpha = np.linalg.norm(W - average_matrix, 2)

    return W, alpha,A

def _gradient(batch_h, batch_y, x,rho):
    m, n = len(batch_h[:, 0]), len(batch_h[0, :])
    y1 = 2 * rho * (batch_h.dot(x)  - batch_y)
    y2 = 2 * rho  + (batch_y - batch_h.dot(x) ) ** 2
    y_d = y1 / y2
    gradient = batch_h.T.dot(y_d)/m


    return gradient
def Hessian(w,X_total,Y_total,rho):
    m_total,n_total = len(X_total[:, 0]),len(X_total[0, :])
    aa = 2*rho-(Y_total - X_total.dot(w)) ** 2
    #aa*=X_total
    bb = ((Y_total - X_total.dot(w) ) ** 2+2*rho)**2
    aa_bb = aa/bb
    X_total1 = np.zeros((m_total,n_total))
    for i in range(m_total):
        X_total1[i,:]=aa_bb[i]*X_total[i,:]
    return (X_total.T.dot(X_total1))/m_total

def f( w, X_total,Y_total,rho):
    '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''


    m_total = len(X_total[:, 0])
    return np.sum(rho * np.log((Y_total- X_total.dot(w)) ** 2 / (2 * rho) + 1)) / m_total
def grad_error(x,pp,X,Y,rho,delta):
    tmp = copy.deepcopy(x)
    dim, n_agent = len(x[:, 0]), len(x[0, :])
    grad_last = np.zeros((dim,n_agent))
    xx = np.zeros((dim, n_agent))
    x_mean = x.mean(axis=1)
    for i in range(n_agent):
        xx[:,i] =x[:,i]- x_mean
        grad_last[:,i] = _gradient( X[i],Y[i],x_mean,rho)
    tmp = tmp - grad_last-pp
    tmp = np.maximum(tmp - delta , 0) - np.maximum(-tmp - delta , 0)
    return np.linalg.norm(x - tmp)+np.linalg.norm(xx)
def SPPDM(x,pp,z,x_last,phi_0,phi_0_inv,X,Y,batch_size,m,AA,BB,alpha,gamma,c,kappa,beta,eta,delta,rho):
    dim, n_agent = len(x[:, 0]), len(x[0, :])
    # ss = x + eta * (x - x_last)
    grad_last = np.zeros((dim, n_agent))

    pp = pp + alpha * x.dot(AA)
    ss = x + eta * (x - x_last)
    tmp = gamma * ss + c * x.dot(BB) + kappa * z - pp



    for i in range(n_agent):
        k_list = np.random.choice(np.arange(m[i]), size=batch_size, replace=False)
        grad_last[:, i] = _gradient(X[i][k_list], Y[i][k_list], ss[:,i],rho)

    tmp -= grad_last
    tmp = tmp.dot(phi_0_inv)
    tmp = np.maximum(tmp - delta * phi_0, 0) - np.maximum(-tmp - delta * phi_0, 0)

    z = z + beta * (tmp - z)



    x_last = x
    x = tmp

    return  x,pp,z,x_last




