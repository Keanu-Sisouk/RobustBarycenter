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
from gudhi.wasserstein import wasserstein_distance
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
    print(Y.shape)
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


def k_means(Y_list, k, its_bar, its_kmeans,order_p, order_q):
    # np.random.seed(3)
    # torch.manual_seed(3)
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = len(Y_list)
    chosen_indices = []
    available_indices = list(range(n_samples))

    # Step 1: Choose the first centroid randomly
    # first_idx = torch.randint(0, n_samples, (1,)).item()
    first_idx = 0
    chosen_indices.append(first_idx)
    centroids = [Y_list[first_idx]]
    available_indices.remove(first_idx)

    # Step 2: Select remaining centroids
    for _ in range(1, k):
        # Compute distances from all available points to the nearest centroid
        available_points = [Y_list[i] for i in available_indices]
        dist_matrix = torch.zeros((len(available_indices), len(centroids))) # (n_samples, k)
        # print(dist_matrix)
        for l in range(len(available_indices)):
            for i in range(len(centroids)):
                Y = Y_list[available_indices[l]].detach().clone().cpu().numpy()
                bary = centroids[i].detach().clone().cpu().numpy()
                dist_matrix[l, i] = wasserstein_distance(Y, bary, order = order_q, internal_p = order_p)
        min_distances = torch.min(dist_matrix, dim=1).values  # shape: (len(available),)
        probs = min_distances ** 2
        probs /= torch.sum(probs)

        # Sample next centroid index from available_indices
        next_local_idx = torch.multinomial(probs, 1).item()
        next_idx = available_indices[next_local_idx]

        centroids.append(Y_list[next_idx])
        chosen_indices.append(next_idx)
        available_indices.remove(next_idx)
    print(chosen_indices)
    print("HALLO")

    # centroids = [Y_list[0], Y_list[2], Y_list[6]] # first and second 
    # centroids = [Y_list[0], Y_list[2], Y_list[7]] # third
    # centroids = [Y_list[0], Y_list[2], Y_list[3]] # fourth and fifth
    # centroids = [Y_list[0], Y_list[5], Y_list[7]] 

    for iteration in range(its_kmeans):
        print("ITERATION: " + str(iteration))
        # Step 1: Assign points to closest centroid
        # print(len(centroids))
        dist_matrix = torch.zeros((n_samples, k)) # (n_samples, k)
        # print(dist_matrix)
        for l in range(n_samples):
            for i in range(k):
                Y = Y_list[l].detach().clone().cpu().numpy()

                bary = centroids[i].detach().clone().cpu().numpy()
                # if l == 0:
                #     print(bary)
                # if iteration == 6:
                #     print( (bary[:, 1] - bary[:, 0]) * 2 ** (1.0 / order_p - 1))
                dist_matrix[l, i] = wasserstein_distance(Y, bary ,matching = False, order = order_q, internal_p = order_p)
        # print(dist_matrix)
        labels = torch.argmin(dist_matrix, dim=1)  # (n_samples,)

        print(labels)
        # Step 2: Recompute centroids
        new_centroids = []
        for i in range(k):
            # cluster_points = Y_list[labels == i]
            cluster_points = [Y_list[j].clone().to(device) for j in range(n_samples) if labels[j] == i]
            # print(len(cluster_points))
            if len(cluster_points) == 0:
                # Reinitialize empty cluster randomly
                new_centroid = Y_list[torch.randint(0, n_samples, (1,)).item()]
            else:
                # new_centroid = barycenter_fn(cluster_points)
                # if iteration == 7 or iteration == 6:
                #     print(cluster_points)
                #     for x in cluster_points:
                #         print(x)
                K = len(cluster_points)
                # print(K)
                cost_list = [lambda x, y: p_norm_q_cost_matrix(x, y, order_p, order_q)] * K
                its = 10
                # X_init = cluster_points[0].clone()
                cluster_points_temp = [x.clone().to(device) for x in cluster_points]
                # n_init = X_init.shape[0]
                # for diag in cluster_points:
                #     X_init, _ = add_projections(X_init, diag, order_p)
                for i in range(K):
                    x = cluster_points_temp[i].clone().to(device)
                    # print(x.shape)
                    for j in range(K):
                        if i == j:
                            continue
                        else:
                            temp, _ = add_projections(x, cluster_points[j], order_p)
                            x = temp.clone()
                    cluster_points_temp[i] = x
                    # print(x.shape)
                # if iteration == 6 or iteration == 7:
                #     print(X_init)
                X_init = cluster_points_temp[0].clone().to(device)
                shape_list = [Y.shape[0] for Y in cluster_points_temp]
                b_list_temp = TT([ot.unif(m) for m in shape_list], device = device)
                # new_centroid, _ = solve_OT_barycenter_fixed_point(
                #     X_init, cluster_points, order_p, b_list, cost_list, lambda y: B(y, order_p, order_q), max_its=its_bar, log=True, stop_threshold=0.)
                new_centroid, _ = solve_OT_barycenter_fixed_point(
                    X_init, cluster_points_temp, b_list_temp, cost_list, lambda y: B(y, order_p, order_q), max_its=its_bar, log=True, stop_threshold=0.)
                # print("NEW BARY")
                # print(new_centroid.shape)

                row_min, _ = new_centroid.min(dim=1, keepdim=True)
                row_max, _ = new_centroid.max(dim=1, keepdim=True)
                is_diag = abs(row_max - row_min) < 1e-4
                new_centroid_without_diag = new_centroid[~is_diag.squeeze()]
                new_centroid_without_diag = new_centroid_without_diag[new_centroid_without_diag[:,1] - new_centroid_without_diag[:,0] >=0]
                # print(new_centroid_without_diag.shape)

                
                # counter = 0
                # counter2 = 0
                # for x in new_centroid:
                #     if x[1] - x[0] < 0:
                #         counter +=1
                # #     if x[1] - x[0] > 1e2:
                # #         counter2 +=1
                # if counter > 0:
                #     print("ERROR 404 POINTS UNDER DIAGONAL")
                #     for i in range(len(cluster_points_temp)):
                #         torch.save(cluster_points_temp[i], "diag"+str(i)+".pt")

                #     print("DIAG SAVED")
                
                # print(counter)
                # print(counter2)
                # counter3 = 0
                # matrix_bool = torch.isnan(new_centroid_without_diag)
                # for x in matrix_bool:
                #     if x[0] == True:
                #         counter3+=1
                #     if x[1] == True:
                #         counter3+=1

                # print(counter3)
            new_centroids.append(new_centroid_without_diag)
        

        # # Step 3: Check convergence
        # shift = torch.norm(new_centroids - centroids, dim=1).sum()
        # if verbose:
        #     print(f"Iter {iteration}: total centroid shift = {shift.item():.6f}")
        # if shift < tol:
        #     break 
        # if iteration == 6 or iteration == 7:
        #     for x in new_centroids:
        #         print(x)

        centroids = [x.clone() for x in new_centroids]
        # print("CENTROIDS COPY")
        # for x in centroids:
        #     print(x)
    return labels, centroids

# labels = k_means(Y_list, k = 3, its_bar = 50, its_kmeans= 3, order_p = 1.3 , order_q = 1.3)
# print(labels)
# labels = k_means(Y_list, k = 3, its_bar = 5, its_kmeans= 5, order_p = 1.3 , order_q = 1.3)
# print(labels)

# its_bar_list = [10, 15, 20]
# its_bar_kmeans = [10, 15, 20]
# order_p_list = [1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1]
# order_q_list = [1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1]


def run_kmeans_grid(Y_list, k, order_p_list, order_q_list, its_bar_list, its_kmeans_list, output_dir="resultsIsabel_trash"):

    os.makedirs(output_dir, exist_ok=True)

    for order_p, order_q, its_bar, its_kmeans in itertools.product(order_p_list, order_q_list, its_bar_list, its_kmeans_list):
        # if order_p == 1.8 or order_p == 1.7:
        #     continue
        labels, centroids = k_means(Y_list, k, its_bar, its_kmeans, order_p, order_q)

        filename = f"labels_p{order_p}_q{order_q}_itsbar{its_bar}_itskmeans{its_kmeans}.pt"
        filepath = os.path.join(output_dir, filename)
        torch.save(labels, filepath)

        print(f"Saved: {filepath}")

        for i in range(len(centroids)):
            filename2 = f"centroid_{i}_for_p{order_p}_q{order_q}_itsbar{its_bar}_itskmeans{its_kmeans}.pt"
            filepath2 = os.path.join(output_dir,filename2)
            torch.save(centroids[i], filepath2)
            print(f"Saved: {filepath2}")
    return 0


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
        X_init, cluster_points_temp, b_list_temp, cost_list, lambda y: B(y, 2, 2), max_its=5, log=True, stop_threshold=0.)
    end = time.time()

    length = end - start

    print(f"Running time b_2 when m = {K}: {length} seconds")