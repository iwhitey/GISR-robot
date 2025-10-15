import torch
from torch.nn.functional import normalize
import numpy as np
import sys

try:
    import graphik
except:
    sys.path.append("../graphIK")
try:
    import liegroups
except:
    sys.path.append("../liegroups")
try:
    import urdfpy
except:
    sys.path.append("../urdfpy")


from liegroups.torch import SE3, SO3, SE3

import graphik
from graphik.utils.roboturdf import load_panda_truncated
from graphik.utils import graph_from_pos, skew


def batch_joint_variables(P, M, num_joints, dim=3, axis_length=1, tol=1e-10):
    """
    Computes joint angles from a batch of 3D points which are assumed to be a
    "complete" set of points that uniquely determine the robot configuration.
    """
    # this assumes all data points in the batch are the same robot
    # P is [batch_size x num_nodes x 3] in canonical ordering

    batch_size = P.shape[0]
    num_nodes = P.shape[1]
    device = M.device

    #____handle non-canonical coordinate system
    x_hat = normalize(P[:,1,:] - P[:,0,:], dim=-1)
    y_hat = normalize(P[:,2,:] - P[:,0,:], dim=-1)
    z_hat = normalize(P[:,3,:] - P[:,0,:], dim=-1)
    R = torch.cat([x_hat.unsqueeze(1),
                      -y_hat.unsqueeze(1),
                      z_hat.unsqueeze(1)],
                     dim = 1).transpose(2,1)

    B = SE3(SO3(R), P[:,0,:])

    #____helper matrices
    omega_z = torch.tensor(skew(np.array([0,0,1.])), dtype=torch.float64, device=device)
    omega_z_sq = torch.matmul(omega_z, omega_z.transpose(1,0))

    #____reduce P by removing x and y nodes for easy indexing
    P = torch.cat([P[:,0].unsqueeze(1), P[:,3:]], dim=1)

    T = torch.eye(dim+1, device=device, dtype=torch.float64).unsqueeze(0).unsqueeze(0).repeat(batch_size,num_joints,1,1)
    theta = torch.zeros([batch_size, num_joints], device=device)

    for idx in range(1, num_joints):

        #____all of this is known a priori
        T_prev = T[:,idx-1].clone() # previous T, .clone() added because backprop fails due to in-place changes
        T_prev_0 = SE3(SO3(M[idx-1,:-1,:-1]), M[idx-1,:-1,-1])
        T_0 = SE3(SO3(M[idx,:-1,:-1]), M[idx,:-1,-1])
        T_0_q = SE3(SO3(M[idx,:-1,:-1]), M[idx,:-1,-1] + M[idx,:-1,-2])
        T_rel = T_prev_0.inv().dot(T_0)
        ps_0 = T_rel.as_matrix()[:-1,-1]
        qs_0 = T_prev_0.inv().dot(T_0_q).as_matrix()[:-1,-1].double()

        #predicted p and q expressed in previous frame
        idx_p = 2*(idx - 1) + 2 # location of idx-th p in P
        idx_q = 2*(idx - 1) + 3 # location of idx-th q in P
        p = B.inv().dot(P[:,idx_p].unsqueeze(1))
        qnorm = P[:,idx_p] + (P[:,idx_q] - P[:, idx_p]) / torch.linalg.norm(P[:,idx_q] - P[:, idx_p], dim=-1).reshape(-1,1)
        q = B.inv().dot(qnorm.unsqueeze(1))

        ps = SO3(T_prev[:,:-1,:-1].transpose(2,1)).dot((p - T_prev[:,:-1,-1]).unsqueeze(1))
        qs = SO3(T_prev[:,:-1,:-1].transpose(2,1)).dot((q - T_prev[:,:-1,-1]).unsqueeze(1))

        if qs.dim()<2:
            qs = qs.unsqueeze(0)
        if ps.dim()<2:
            ps = ps.unsqueeze(0)
        A_ = torch.matmul(qs_0,omega_z)
        B_ = torch.matmul(qs_0,omega_z_sq)

        theta[:, idx-1] = torch.atan2(
            -torch.bmm(A_.reshape(1,1,-1).expand(batch_size,-1,-1), qs.unsqueeze(1).transpose(2,1)).squeeze(1),
            torch.bmm(B_.reshape(1,1,-1).expand(batch_size,-1,-1), qs.unsqueeze(1).transpose(2,1)).squeeze(1)
        ).reshape(-1)
        
        if theta.shape[0] == 1:
            rotz = SO3.rotz(theta[:, idx-1].unsqueeze(0).clone()) # .clone() added on theta since backprop fails due to in-place changes
            rotmat = SE3(rotz, torch.zeros([batch_size,3], device=device, dtype=torch.float64))
            T[:,idx] = torch.bmm(torch.bmm(T_prev, rotmat.as_matrix().unsqueeze(0)), T_rel.as_matrix().double().expand(batch_size,-1,-1))
        else:
            rotz = SO3.rotz(theta[:, idx-1].clone()) # .clone() added on theta since backprop fails due to in-place changes
            rotmat = SE3(rotz, torch.zeros([batch_size,3], device=device, dtype=torch.float64))
            T[:,idx] = torch.bmm(torch.bmm(T_prev, rotmat.as_matrix()), T_rel.as_matrix().double().expand(batch_size,-1,-1))

    return theta


def get_constant_data(robot, device='cuda'):
    # num_joints = robot.n
    # num_nodes = robot.n + 1

    #______get zero config poses of all joints
    M = torch.zeros([len(robot), robot.dim + 1, robot.dim + 1], device=device)
    for idx in range(len(robot)):
        M[idx] = torch.tensor([robot.pose(robot.zero_configuration(), list(robot.nodes())[idx]).as_matrix()], dtype=torch.float)

    #______make copy of graph representing robot (to avoid accidentally modifying the object)
    G = robot.to_directed()

    #______change relative transforms between joints to numpy matrices
    for u, v in G.edges():
        if "T" in G[u][v]:
            G[u][v]["T"] = G[u][v]["T"].as_matrix()

    S_ = []
    #______delete redundant data from GraphIK
    for node, data in G.nodes(data=True):
        for att in list(data.keys()):
            if att in ["T0", "lb", "ub"]:
                del G.nodes[node][att]
            if att=="S":
                S_+=[torch.from_numpy(G.nodes[node][att])]

    S = torch.cat(S_[:-1]).float().unsqueeze(0)

    return M, S


def solve_mds(distance_mat, num_dimensions=3):
    """
    Solve classical Multidimensional Scaling (MDS) problem of finding 
    3D points from a batch of euclidean distance matrices.
    """

    num_points = distance_mat.shape[1]
    ones = torch.ones((num_points, 1), dtype=torch.float32, device='cuda')
    eye = torch.eye(num_points, dtype=torch.float32, device='cuda')

    # geometric centerig -> centroid of the point set is its origin
    geo_centering = eye - ((1 / num_points) * ones @ ones.T)

    # calculate gram matrix G from distance matrix
    G = -0.5 * (geo_centering @ distance_mat @ geo_centering)

    # eigenvalue decomposition of G to obtain eigenvalues L and eigenvectors Q
    # L, Q = torch.symeig(G, eigenvectors=True) # torch 1.4.0
    L, Q = torch.linalg.eigh(G)
    # assert L[L >= -1e-2].shape[0] == 16, "ERROR: G_opt is not PSD!"

    # EVD(G) should be close to G
    # rtol = 1e-2; atol=1e-8
    # if not torch.allclose(Q @ torch.diag_embed(L) @ Q.transpose(-1, -2), G, rtol=rtol, atol=atol, equal_nan=False):
    #     print(f"|EVD(G) - G| condition failed for rtol={rtol}, atol={atol}")
    
    # take last 3 (largest) eigenvalues/vectors (eigenvals L are in ascending order)
    Q_3d = Q[:, :, -3:]
    L_3d = torch.sqrt(1e-4 + L[:, -3:])
    assert not torch.any(torch.isnan(L_3d)), f"ERROR: {torch.sum(torch.isnan(L_3d))} NaNs found in sqrt(eigenvals)"
    # if torch.any(torch.isnan(L_3d)): print(f"ERROR: {torch.sum(torch.isnan(L_3d))} NaNs found in sqrt(eigenvals)")

    X_3d = Q_3d @ torch.diag_embed(L_3d) # (B, N, d) @ (B, d, d) = (B, N, d)
    return X_3d



def joint_angles_from_complete_point_set(P):
    """
    Finds joint angles from a complete set of 3D points using GraphIK.
    """
    robot, graph = load_panda_truncated(limits=None)
    G_sol = graph_from_pos(P, graph.node_ids)
    q_sol = graph.joint_variables(G_sol)
    return np.array(list(q_sol.values()))


def points_from_gram_eig_decomposition(G):
    """
    Finds 3D points from a Gram matrix using eigendecomposition.
    """
    #eigenvalue decomposition of G to obtain eigenvalues L and eigenvectors Q
    L, Q = torch.symeig(G, eigenvectors=True) #torch 1.4.0
    # L, Q = torch.linalg.eigh(G)

    # EVD(G) should be close to G
    # rtol = 1e-2; atol=1e-8
    # if not torch.allclose(Q @ torch.diag_embed(L) @ Q.transpose(-1, -2), G, rtol=rtol, atol=atol, equal_nan=False):
    #     print(f"|EVD(G) - G| condition failed for rtol={rtol}, atol={atol}")
    
    # take last 3 (largest) eigenvalues/vectors (eigenvals L are in ascending order)
    Q_3d = Q[:, :, -3:]
    L_3d = torch.sqrt(1e-4 + L[:, -3:])
    assert not torch.any(torch.isnan(L_3d)), f"ERROR: {torch.sum(torch.isnan(L_3d))} NaNs found in sqrt(eigenvals)"
    # if torch.any(torch.isnan(L_3d)): print(f"ERROR: {torch.sum(torch.isnan(L_3d))} NaNs found in sqrt(eigenvals)")

    X_3d = Q_3d @ torch.diag_embed(L_3d) # (B, N, d) @ (B, d, d) = (B, N, d)
    return X_3d


def upper_triu_from_symmetric(X):
    """
    Extracts the upper triangular values from a symmetric EDM matrix.
    """
    X_triu = torch.triu(X)
    x, y  = torch.triu_indices(X.shape[1], X.shape[2], offset=1, dtype=torch.long, device='cuda')    
    X_upper_triu = X_triu[:, x, y]
    return X_upper_triu


def symmetric_from_upper_triu_values(X, N):
    """
    Creates a complete euclidean distance matrix from the upper triangular values.
    """
    if isinstance(X, np.ndarray):
        X_symmetric = np.zeros((X.shape[0], N, N))
        x_idxs = np.arange(0, N-1)
        y_idxs = np.arange(1, N)
    else:
        X_symmetric = torch.zeros(X.shape[0], N, N, dtype=torch.float32, device='cuda')
        x_idxs = torch.arange(0, N-1, device='cuda')
        y_idxs = torch.arange(1, N, device='cuda')

    start = 0
    offset = N-1

    for x_idx in x_idxs:
        X_symmetric[:, x_idx, y_idxs[x_idx]:] = X[:, start:start+offset]
        start += offset
        offset -= 1

    X_symmetric = X_symmetric + X_symmetric.transpose(-1, -2)
    assert torch.equal(X_symmetric.squeeze(0).transpose(-1, -2), X_symmetric.squeeze(0)), "symmetric_from_upper_triu_values: X_symmetric fails symmetry check!"
    return X_symmetric
