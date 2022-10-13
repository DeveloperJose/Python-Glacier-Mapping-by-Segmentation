import multiprocessing

import numpy as np
import numba

@numba.njit()
def get_neighbors(im, coords):
    r, c = coords
    possible = [(r-1, c-1), # bot-left
            (r-1, c), # down
            (r-1, c+1), # bot-right
            (r, c+1), # right
            (r+1, c+1), # top-right
            (r+1, c), # up
            (r+1, c-1), # top-left
            (r, c-1) # left
    ]

    real = []
    for tup in possible:
        tr, tc = tup
        if tr >= 0 and tr < im.shape[0] and tc >= 0 and tc < im.shape[1]:
            real.append(tup)

    return real

@numba.njit()
def breadth_first_search(im, source):
    visited = set([source])
    prev = np.zeros((im.shape[0], im.shape[1], 2))
    prev[:, :] = -1

    Q = [source]

    # min_val = im.min()

    best_u = source
    best_elev = im[source]

    while len(Q)>0:
        u = Q.pop(0)
        curr_elev = im[u]

        # if abs(curr_elev - min_val) < 0.01:
        #     return prev, u

        if curr_elev < best_elev:
            best_elev = curr_elev
            best_u = u
            
        for v in get_neighbors(im, u):
            # Water can only flow down
            neigh_elev = im[v]
            if v not in visited and neigh_elev < curr_elev:
                prev[v] = u
                visited.add(v)
                Q.append(v)
    return prev, best_u

@numba.njit()
def get_path(prev,v):
    prev_tuple = prev[v]
    if prev_tuple.sum()<0:   # v is the origin
        return [v]
    return get_path(prev, (int(prev_tuple[0]), int(prev_tuple[1]))) + [v]

# @numba.njit()
def process(im_elevation, tup):
    u, v = tup
    prev, goal = breadth_first_search(im_elevation, (u, v))
    path = get_path(prev, goal)
    return path

# @numba.njit()
def paths_to_im(all_paths, shape):
    water = np.ones((shape[0], shape[1]))
    for path in all_paths:
        if path is None:
            continue
        contribution = np.linspace(1 / len(path), 0, len(path))
        for idx, (r, c) in enumerate(path):
            water[r, c] += contribution[idx]
    return water


def augment(im_elevation):
    pairs = []
    for p in np.mgrid[0:im_elevation.shape[0]:50j, 0:im_elevation.shape[1]:50j].reshape(2,-1).T:
        pairs.append((int(p[0]), int(p[1])))
    all_paths = []
    for p in pairs:
        all_paths.append(process(im_elevation, p))
    return paths_to_im(all_paths, im_elevation.shape)