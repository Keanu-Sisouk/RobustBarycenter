import os
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
from ot_bar.utils import TT, TN
from ot_bar.solvers2 import solve_OT_barycenter_fixed_point, StoppingCriterionReached
from ot_bar.solvers2 import add_projections
import torch
from torch.optim import Adam
from matplotlib import cm
from tqdm import tqdm
# from gudhi.wasserstein import wasserstein_distance
import itertools
import time

np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)
colours = ['#7ED321', '#4A90E2', '#9013FE']

d = 2
# n = 400

tempListMax = [np.load("IsabelMinThreshNew"+str(i)+".npy") for i in range(0,12)]
# tempListMin = [np.load("")]

# indices_list_for_input = [0,3,4,5,6,7,11] #first
# indices_list_for_input = [0,3,4,5,6,7,8,11] # second
# indices_list_for_input = [0,1,2,3,5,6,7,10,11] #third
# indices_list_for_input = [0,3,4,10,11] #fourth
# indices_list_for_input = [1,2,3,4,7,10,11] #fifth
# indices_list_for_input = [0,1,3,4,5,6,9]
indices_list_for_input = [0,1,2,3,4,5,6,7,8,9,10,11]



Y_list = TT([np.load("IsabelMinThreshNew"+str(i)+".npy") for i in indices_list_for_input], device=device)
for Y in Y_list:
    Y = Y[Y[:,1]-Y[:,0] > 0.01*(Y[0,1]-Y[0,0])]
    # print(Y.shape)
m_list = [Y.shape[0] for Y in Y_list]
b_list = TT([ot.unif(m) for m in m_list], device = device)
# a = TT(ot.unif(n), device = device)


def p_norm_q_cost_matrix(u, v, order_p, order_q):
    dist= torch.sum(torch.abs(u[:, None, :] - v[None, :, :])**order_p, axis=-1)**(order_q / order_p)

    is_diag_u = torch.all(u == u[:, :1], dim = 1)
    is_diag_v = torch.all(v == v[:, :1], dim = 1)

    diag_mask = is_diag_u[:, None] & is_diag_v[None, :]

    dist[diag_mask] = 0.0

    return dist

# fonction B alternative qui n'utilise pas pytorch, seulement pour p=2
def B2(y, order_p, order_q, its=1050, lr=2, log=False, stop_threshold=1e-20):
    """
    Computes the barycenter images for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    Output: (n, d) array
    """
    x = torch.zeros_like(y[0])
    #x = torch.randn(n, d, device=device, dtype=torch.double)
    #x.requires_grad_(True)
    loss_list = [1e10]
    #opt = Adam([x], lr=lr)
    exit_status = 'unknown'
    try:
        for _ in range(its):
            #opt.zero_grad()
            loss = torch.sum(C(x, y, order_p, order_q))
            #loss.backward()
            #opt.step()
            z = 0.
            m = torch.zeros_like(y[0])
            for k in range(len(y)):
                dk = torch.linalg.norm(y[k] - x)**(order_p-order_q)
                m += 1/len(y) * y[k]  / dk
                z += 1/len(y) * 1/ dk
            x = m / z   

            # print(loss)
            loss_list.append(loss.item())
            if stop_threshold > loss_list[-2] - loss_list[-1] >= 0:
                exit_status = 'Local optimum'
                raise StoppingCriterionReached
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached
    except StoppingCriterionReached:
        if log:
            return x, {'loss_list': loss_list[1:], 'exit_status': exit_status}
        return x





def B(y, order_p, order_q, its=250, lr=1, log=False, stop_threshold=1e-20):
    """
    Computes the barycenter images for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    Output: (n, d) array
    """

    n = y[0].shape[0]
    x = torch.randn(n, d, device=device, dtype=torch.double)
    x.requires_grad_(True)
    loss_list = [1e10]
    opt = Adam([x], lr=lr)
    exit_status = 'unknown'
    try:
        for _ in range(its):
            opt.zero_grad()
            loss = torch.sum(C(x, y, order_p, order_q))
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
            if stop_threshold > loss_list[-2] - loss_list[-1] >= 0:
                exit_status = 'Local optimum'
                raise StoppingCriterionReached
        exit_status = 'Max iterations reached'
        raise StoppingCriterionReached
    except StoppingCriterionReached:
        if log:
            return x, {'loss_list': loss_list[1:], 'exit_status': exit_status}
        return x


def B_analytic(y, order_p, order_q, its=250, lr=1, log=False, stop_threshold=1e-20):
    x = torch.zeros_like(y[0], device=device, dtype=torch.double)
    x.requires_grad_(True)
    K = len(y)
    for k in range(K):
        x = x + (1/K)*y[k]
    return x


def C(x, y, order_p, order_q):
    """
    Computes the ground barycenter cost for candidate points x (n, d) and
    measure supports y: List(n, d_k).
    """
    n = x.shape[0]
    # print(x.shape)
    # print(y[0].shape)
    K = len(y)
    out = torch.zeros(n, device=device)
    for k in range(K):
        # print(y[k].shape)
        # print(x.shape)
        # out += (1 / K) * torch.sum(torch.abs(x - y[k])**order_p, axis=1)**(order_q / order_p)
        x_diag = torch.all(x == x[:, [0]], dim=1)            # shape: (n,)
        y_diag = torch.all(y[k] == y[k][:, [0]], dim=1)       # shape: (n,)
        both_diag = x_diag & y_diag                          # shape: (n,)

        # Compute the full cost
        cost = torch.sum(torch.abs(x - y[k])**order_p, axis=1)**(order_q / order_p)

        # Set cost to zero where both x and y[k] are diagonal
        cost[both_diag] = 0.0

        # Accumulate the weighted cost
        out += (1 / K) * cost
    return out



# X_init = torch.rand(n, d, device=device, dtype=torch.double)
# %% fixed-point for (p, q) = (1.5, 1.5)
p, q = 1.5, 1.5
# K = 4
# cost_list = [lambda x, y: p_norm_q_cost_matrix(x, y, p, q)] * K
its = 10

K_number = [4,6]
for K in K_number:
    s = 0
    if K == 6:
        s = 6
    else:
        s = 8

    cluster_points = [Y_list[j].clone().to(device) for j in range(s,12)]

    cluster_points_temp = [x.clone().to(device) for x in cluster_points]
    cost_list = [lambda x, y: p_norm_q_cost_matrix(x, y, 2, 2)] * K
    for i in range(K):
        x = cluster_points_temp[i].clone().to(device)
        # print(x.shape)
        for j in range(K):
            if i == j:
                continue
            else:
                temp, _ = add_projections(x, cluster_points_temp[j], 2)
                x = temp.clone()
        cluster_points_temp[i] = x
        # print(x.shape)
    start = time.time()
    X_init = cluster_points_temp[0].clone().to(device)
    shape_list = [Y.shape[0] for Y in cluster_points_temp]
    b_list_temp = TT([ot.unif(m) for m in shape_list], device = device)
    # new_centroid, _ = solve_OT_barycenter_fixed_point(
    #     X_init, cluster_points, order_p, b_list, cost_list, lambda y: B(y, order_p, order_q), max_its=its_bar, log=True, stop_threshold=0.)
    new_centroid, _ = solve_OT_barycenter_fixed_point(
        X_init, cluster_points_temp, b_list_temp, cost_list, lambda y: B_analytic(y, 2, 2), max_its=5, log=True, stop_threshold=0.)
    end = time.time()

    length = end - start

    print(f"Running time arithmetic mean when m = {K}: {length} seconds")