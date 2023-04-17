import collections

import numba
import numpy as np
from PIL import Image
from tqdm import tqdm


@numba.njit()
def get_neighbors(im, coords):
    r, c = coords
    possible = [(r-1, c-1),  # bot-left
                (r-1, c),  # down
                (r-1, c+1),  # bot-right
                (r, c+1),  # right
                (r+1, c+1),  # top-right
                (r+1, c),  # up
                (r+1, c-1),  # top-left
                (r, c-1)  # left
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
        return [v]
    return get_path(prev, (int(prev_tuple[0]), int(prev_tuple[1]))) + [v]


@numba.njit()
def manhattan_cost(u, v):
    rr = abs(u[0] - v[0])
    cc = abs(u[1] - v[1])
    return (rr + cc)


def breadth_first_search_v2(im, water_allpath, source):
    visited = set([source])

    Q = collections.deque()
    Q.append(source)

    while len(Q) > 0:
        u = Q.popleft()
        curr_elev = im[u]

        # Only accumulate water in neighbors
        if u != source:
            water_allpath[u] += 1  # manhattan_cost(u, source)

        # Get only valid neighbors
        for v in get_neighbors(im, u):
            neigh_elev = im[v]
            # Visit if not visited and if elevation is lower as water can only flow down
            if v not in visited and neigh_elev < curr_elev:
                visited.add(v)
                Q.append(v)

    return water_allpath


@numba.njit()
def mean_std(im):
    return (im - im.mean()) / im.std()


@numba.njit()
def min_max(im):
    return (im - im.min()) / im.max()


def resize(arr, new_rows, new_cols):
    arr = min_max(arr)*255
    arr = arr.astype(np.uint8).reshape(arr.shape[0], arr.shape[1])
    arr = np.array(Image.fromarray(arr).resize((new_cols, new_rows)))  # , Image.LANCZOS))
    arr = arr.astype(np.float32) / 255.0
    return arr


def compute_phys_v2(elevation, res=64, scale=0.3):
    # Downsize elevation map for faster computation. Also normalizes to [0,1]
    original_shape = elevation.shape
    if scale != 1:
        new_rows = int(elevation.shape[0] * scale)
        new_cols = int(elevation.shape[1] * scale)
        elevation = resize(elevation, new_rows, new_cols)
    # print(f'From {original_shape} to {elevation.shape}, {elevation.min()}, {elevation.max()}, {elevation.dtype}')

    # Create an empty image where the water accumulation will be stored
    water_allpath = np.zeros((elevation.shape[0], elevation.shape[1]), dtype=np.float32)

    # Create the pairs that will be explored by BFS
    pairs = []
    step = 1 if res == "full" else complex(0, res)
    for p in np.mgrid[0:elevation.shape[0]-1:step, 0:elevation.shape[1]-1:step].reshape(2, -1).T:
        pairs.append((int(p[0]), int(p[1])))

    # Perform BFS
    # for u,v in tqdm(pairs, position=2):
    for u, v in pairs:
        water_allpath = breadth_first_search_v2(elevation, water_allpath, (u, v))

    # Resize back to original elevation map size
    if scale != 1:
        water_allpath = resize(water_allpath, original_shape[0], original_shape[1])

    # Expand last dimension for concatenation with other channels
    return water_allpath.reshape(original_shape[0], original_shape[1], 1)


if __name__ == '__main__':
    import slice as fn

    dem = fn.read_tiff('/data/baryal/HKH/DEM/image1.tif')
    dem_np = np.transpose(dem.read(), (1, 2, 0)).astype(np.float32)
    dem_np = np.nan_to_num(dem_np)
    elevation = dem_np[:, :, 0][:, :, None]

    phys_output = compute_phys_v2(elevation, res=64)
    print(phys_output.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(phys_output, cmap='gray')
    plt.savefig('physics_example.png')
