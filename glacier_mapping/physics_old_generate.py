import pathlib
import multiprocessing
import collections

import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer

import numba


rng = np.random.default_rng(41)

# version1
# 128 takes 2350.16sec = 39min
# 256 takes 9452.53sec = 157min = 2.62hr

# version2
# 64 takes 334.28sec
# 128 takes 1607.37sec
# 256 takes 8970.83sec

# version3
# 64 takes 351.79sec


@numba.njit()
def get_neighbors(im, coords):
    r, c = coords
    possible = [
        (r - 1, c - 1),  # bot-left
        (r - 1, c),  # down
        (r - 1, c + 1),  # bot-right
        (r, c + 1),  # right
        (r + 1, c + 1),  # top-right
        (r + 1, c),  # up
        (r + 1, c - 1),  # top-left
        (r, c - 1),  # left
    ]

    real = []
    for tup in possible:
        tr, tc = tup
        if tr >= 0 and tr < im.shape[0] and tc >= 0 and tc < im.shape[1]:
            real.append(tup)

    return real


@numba.njit()
def get_path(prev, v):
    prev_tuple = prev[v]
    if prev_tuple.sum() < 0:  # v is the origin
        # L = numba.typed.List()
        L = []
        L.append(v)
        return L
    L = get_path(prev, (int(prev_tuple[0]), int(prev_tuple[1])))
    L.append(v)
    return L


@numba.jit(forceobj=True, fastmath=True)
def breadth_first_search_v1(im, source, use_1path=True):
    visited = set([source])
    prev = np.zeros((im.shape[0], im.shape[1], 2))
    prev[:, :] = -1

    # Q = [source]

    Q = collections.deque()
    Q.append(source)

    # Q = numba.typed.List()
    # Q.append(source)

    min_val = im.min()
    best_u = source
    best_elev = im[source]
    # water_allpath = np.zeros((im.shape[0], im.shape[1]))

    iters = 0
    while len(Q) > 0:
        iters += 1
        # u = Q.pop(0)
        u = Q.popleft()
        curr_elev = im[u]

        # Keep track of best so far in case we stop early
        if curr_elev < best_elev:
            best_elev = curr_elev
            best_u = u

        # Early goal
        if source != u and abs(curr_elev - min_val) <= 1e-3:
            return prev, best_u

        # Max iterations
        if iters > 512:  # 2^10=1024 | 2^16=65536 -> explored N levels
            # print(f'Max iters for BFS: best_u={best_u} with elevation={im[best_u]} for source={source} with elevation={im[source]}')
            # print('Max iters for BFS')
            return prev, best_u

        # Get only valid neighbors
        for v in get_neighbors(im, u):
            neigh_elev = im[v]
            # Visit if not visited and if elevation is lower as water can only flow down
            if v not in visited and neigh_elev < curr_elev:
                prev[v] = u
                visited.add(v)
                Q.append(v)

    return prev, best_u
    # if use_1path:
    #     water_1path = np.zeros((im.shape[0], im.shape[1]))
    #     path = get_path(prev, best_u)
    #     contribution = np.linspace(0, 1 / (len(path) / 2), len(path))
    #     for idx, (r, c) in enumerate(path):
    #         water_1path[r, c] += contribution[idx]
    #     # Mean-STD normalization
    #     water_1path = (water_1path - water_1path.mean()) / water_1path.std()
    #     return water_1path
    # else:
    #     raise Exception('Not implemented')
    # return prev, best_u


# @numba.njit(fastmath=True)
# def get_pairs(im):
#     # ind = np.argpartition(im.ravel(), -32768)[-32768:]
#     # ind = np.argsort(im.ravel())[-1024:]
#     # pairs = []
#     # # rows, cols = np.unravel_index(ind, im.shape)
#     # rows = ind // im.shape[1]
#     # cols = ind % im.shape[1]
#     # for r, c in zip(rows, cols):
#     #     pairs.append((r, c))
#     return pairs


@numba.jit(forceobj=True, fastmath=True)
def breadth_first_search_v2(im, water_allpath, source):
    visited = set([source])
    prev = np.zeros((im.shape[0], im.shape[1], 2))
    prev[:, :] = -1

    Q = collections.deque()
    Q.append(source)

    min_val = im.min()
    best_u = source
    best_elev = im[source]

    iters = 0
    while len(Q) > 0:
        iters += 1
        u = Q.popleft()
        curr_elev = im[u]
        im[u] += 0.01
        water_allpath[u] += 1

        # Keep track of best so far in case we stop early
        if curr_elev < best_elev:
            best_elev = curr_elev
            best_u = u

        # Early goal
        if source != u and abs(curr_elev - min_val) <= 1e-3:
            return im, water_allpath, prev, best_u

        # Max iterations
        if iters > 512:  # 2^10=1024 | 2^16=65536 -> explored N levels
            return im, water_allpath, prev, best_u

        # Get only valid neighbors
        for v in get_neighbors(im, u):
            neigh_elev = im[v]
            # Visit if not visited and if elevation is lower as water can only flow down
            if v not in visited and neigh_elev < curr_elev:
                prev[v] = u
                visited.add(v)
                Q.append(v)

    return im, water_allpath, prev, best_u


@numba.jit(forceobj=True, fastmath=True)
def get_water_im(shape, all_paths):
    water_1path = np.zeros((shape[0], shape[1]))

    for path in all_paths:
        contribution = np.linspace(0, 1 / (len(path) / 2), len(path))
        for idx, (r, c) in enumerate(path):
            # print(f'u={u}, v={v}, path={path}, r={r}, c={c}, {len(path)}')
            water_1path[r, c] += contribution[idx]

    # Mean-STD normalization
    mu = water_1path.mean()
    std = water_1path.std()
    if std <= 1e-10:
        # print(f'mu={mu}, std={std} for a sample')
        print("std <= 1e-10 for a sample")
    else:
        water_1path = (water_1path - mu) / std

    return water_1path


@numba.njit()
def mean_std(im):
    return (im - im.mean()) / im.std()


@numba.njit()
def min_max(im):
    return (im - im.min()) / im.max()


if __name__ == "__main__":
    input_path = pathlib.Path("/home/jperez/data/HKH/processed_L07_2005")
    # output_path = pathlib.Path('/home/jperez/programming/glacial_phys_data/cleanice')
    output_path = input_path

    assert input_path.exists()
    assert output_path.exists()

    for folder in ["train", "val", "test"]:
        assert (input_path / folder).exists()

    def process(filename: pathlib.Path):
        data = np.load(filename)
        im_elevation = data[:, :, 8]

        # %% v2 and v3
        water = min_max(im_elevation)
        water_allpath = np.zeros((water.shape[0], water.shape[1]), dtype=np.float32)

        # %% pairs
        # pairs = [(u, v) for u in range(im_band.shape[0]) for v in range(im_band.shape[1])]
        pairs = []
        for p in (
            np.mgrid[
                0 : im_elevation.shape[0] - 1 : 64j, 0 : im_elevation.shape[1] - 1 : 64j
            ]
            .reshape(2, -1)
            .T
        ):
            pairs.append((int(p[0]), int(p[1])))
        # pairs = get_pairs(im_elevation)
        # all_paths = numba.typed.List([(0, 0), (0, 1)])
        # all_paths.pop()

        # %% v1
        # all_paths = []
        # for u, v in tqdm(pairs, desc=f'{filename}', position=2):
        #     prev, goal = breadth_first_search(im_elevation, (u, v), use_1path=True)
        #     path = get_path(prev, goal)
        #     if len(path) > 1:
        #         all_paths.append(path)

        # water_1path = get_water_im(im_elevation.shape, all_paths)

        # %% v2
        # for u, v in tqdm(pairs, desc=f'{filename}', position=2):
        #     water, water_allpath, _, _ = breadth_first_search_v2(water, water_allpath, (u, v))
        # water_allpath = mean_std(water_allpath)

        # %% v3
        for u, v in tqdm(pairs, desc=f"{filename}", position=2):
            prev, goal = breadth_first_search_v1(water, (u, v))
            path = get_path(prev, goal)
            contribution = np.linspace(0, 1 / (len(path) / 2), len(path))
            water_level = np.linspace(0, 0.01, len(path))
            for idx, (r, c) in enumerate(path):
                water_allpath[r, c] += contribution[idx]
                water[r, c] += water_level[idx]

        water_allpath = mean_std(water_allpath)
        # %% output
        folder = filename.parent.name  # [train, val, test]
        filename = str(filename).replace("tiff", "physics_v3_w0.01_64")

        # v1
        # np.save(output_path / folder / filename, water_1path)

        # v2
        np.save(output_path / folder / filename, water_allpath)

    # print(f'Compiling functions')
    # test = np.zeros((1, 1))
    # test_neigh = get_neighbors(test, (0, 0))
    # test_prev, test_u = breadth_first_search_v1(test, (0, 0), use_1path=True)
    # test_water, test_water_allpath, test_prev, test_best_u = breadth_first_search_v2(test, test, (0, 0))
    # test_path = get_path(test_prev, test_u)
    # test_pairs = get_pairs(test)

    print(f"Generating physics images for {input_path}")
    start_time = timer()
    for dataset in ["train", "val", "test"]:
        files = (input_path / dataset).glob("*tiff*")
        files = [pathlib.Path(fname) for fname in files]

        pbar = tqdm(total=len(files), desc=f"Processing dataset {dataset}")

        # for file in files:
        #     process(file)
        #     pbar.update(1)

        with multiprocessing.Pool(32) as pool:
            for result in pool.imap_unordered(process, files):
                pbar.update(1)

        pbar.close()

    duration = timer() - start_time
    print(f"Whole program finished in {duration:.2f}sec!")
    # water = np.ones((im_band.shape[0], im_band.shape[1]))
    # for path in all_paths:
    #     if path is None:
    #         continue
    #     contribution = np.linspace(0, 1 / (len(path) / 2), len(path))
    #     for idx, (r, c) in enumerate(path):
    #         water[r, c] += contribution[idx]

    # np.save('water.npy', water)
    # plt.figure()
    # plt.imshow(water)
    # plt.savefig('water.png')
