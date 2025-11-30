import numpy as np
import numba

# ------------------------------------------------------------
# Static neighbor offsets (Numba-optimized)
# ------------------------------------------------------------
neighbor_offsets = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ],
    dtype=np.int32,
)


# ------------------------------------------------------------
# Optimized BFS for flow accumulation
# ------------------------------------------------------------
@numba.njit()
def bfs_fast(im, water_allpath, source, queue_r, queue_c, visited):
    """
    Single-source BFS over a DEM surface.

    im            : 2D elevation array (float32)
    water_allpath : 2D accumulation array (float32, updated in-place)
    source        : (row, col) tuple
    queue_r/c     : preallocated integer arrays of length rows * cols
    visited       : 2D bool mask (reset at entry)
    """
    rows, cols = im.shape

    # Reset visited mask (Numba-friendly looping)
    for i in range(rows):
        for j in range(cols):
            visited[i, j] = False

    sr, sc = source

    # Initialize BFS queue
    queue_start = 0
    queue_end = 1
    queue_r[0] = sr
    queue_c[0] = sc
    visited[sr, sc] = True

    while queue_start < queue_end:
        ur = queue_r[queue_start]
        uc = queue_c[queue_start]
        queue_start += 1

        curr_elev = im[ur, uc]

        # Every visited cell except source gets +1 to accumulation
        if not (ur == sr and uc == sc):
            water_allpath[ur, uc] += 1.0

        # Loop over 8 neighbors using static offsets
        for k in range(8):
            vr = ur + neighbor_offsets[k, 0]
            vc = uc + neighbor_offsets[k, 1]

            # Bounds check
            if vr < 0 or vr >= rows or vc < 0 or vc >= cols:
                continue

            # Downhill rule: strictly lower elevation
            if not visited[vr, vc] and im[vr, vc] < curr_elev:
                visited[vr, vc] = True
                queue_r[queue_end] = vr
                queue_c[queue_end] = vc
                queue_end += 1

    return water_allpath


# ------------------------------------------------------------
# Resize helper (cv2 only, preserves raw numeric scale)
# ------------------------------------------------------------
def resize(arr: np.ndarray, new_rows: int, new_cols: int) -> np.ndarray:
    """
    Resize a 2D float32 array to (new_rows, new_cols), preserving
    the original numeric scale (no normalization).

    Requires OpenCV (cv2).
    """
    import cv2  # type: ignore

    arr = arr.astype(np.float32, copy=False)
    resized = cv2.resize(arr, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


# ------------------------------------------------------------
# Flow accumulation from elevation
# ------------------------------------------------------------
def compute_flow(elevation, res=64, scale=0.3):
    """
    Compute a simple flow-accumulation field via BFS over elevation.

    Parameters
    ----------
    elevation : np.ndarray
        Either (H, W) or (H, W, C). If 3D, only channel 0 is used
        as elevation. Values are assumed to be in meters (raw).
    res : int or "full"
        Sampling step for BFS sources. "full" => 1 (every pixel).
    scale : float
        Optional downscale factor to accelerate BFS. Flow is then
        upscaled back to original resolution.

    Returns
    -------
    flow : np.ndarray
        Raw flow accumulation array of shape (H, W), float32.
    """
    # Extract 2D elevation field
    if elevation.ndim == 3:
        elev_2d = elevation[:, :, 0].astype(np.float32)
    else:
        elev_2d = elevation.astype(np.float32)

    original_shape = elev_2d.shape

    # Downscale for speed if needed
    if scale != 1.0:
        new_rows = int(original_shape[0] * scale)
        new_cols = int(original_shape[1] * scale)
        elev_2d = resize(elev_2d, new_rows, new_cols)

    rows, cols = elev_2d.shape

    # Accumulation map
    water_allpath = np.zeros((rows, cols), dtype=np.float32)

    # Allocate BFS buffers ONCE
    queue_r = np.zeros(rows * cols, dtype=np.int32)
    queue_c = np.zeros(rows * cols, dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=np.bool_)

    # Interpret "full" as res = 1 (every pixel as a source)
    if res == "full":
        res = 1

    step = int(res)

    for i in range(0, rows, step):
        for j in range(0, cols, step):
            bfs_fast(elev_2d, water_allpath, (i, j), queue_r, queue_c, visited)

    # Upscale back to original resolution if we downscaled
    if scale != 1.0:
        water_allpath = resize(water_allpath, original_shape[0], original_shape[1])

    return water_allpath.astype(np.float32)


# ------------------------------------------------------------
# Numba-based uniform filter (for TPI)
# ------------------------------------------------------------
@numba.njit()
def uniform_filter_numba(arr, radius):
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    half = radius // 2

    for i in range(rows):
        for j in range(cols):
            total = 0.0
            count = 0
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        total += arr[ni, nj]
                        count += 1
            result[i, j] = total / count
    return result


def compute_tpi(elevation: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Topographic Position Index (TPI):
        TPI = elevation - local_mean(elevation)

    Returns RAW TPI values (can be positive or negative),
    with no normalization applied.

    Parameters
    ----------
    elevation : np.ndarray
        2D elevation array (H, W).
    radius : int
        Window "diameter" for local mean.

    Returns
    -------
    tpi : np.ndarray
        (H, W) float32 array of TPI.
    """
    elevation = elevation.astype(np.float32, copy=False)

    try:
        mean_elev = uniform_filter_numba(elevation, radius)
    except Exception:
        from scipy.ndimage import uniform_filter  # type: ignore

        mean_elev = uniform_filter(elevation, size=radius)

    tpi = elevation - mean_elev
    return tpi.astype(np.float32)


# ------------------------------------------------------------
# Numba-based rolling std (for roughness)
# ------------------------------------------------------------
@numba.njit()
def rolling_std_numba(arr, window):
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    half = window // 2

    padded = np.zeros((rows + 2 * half, cols + 2 * half), dtype=arr.dtype)
    padded[half : half + rows, half : half + cols] = arr

    for i in range(rows):
        for j in range(cols):
            # Extract local window
            wsum = 0.0
            count = 0
            for di in range(window):
                for dj in range(window):
                    v = padded[i + di, j + dj]
                    wsum += v
                    count += 1
            mean = wsum / count

            var = 0.0
            for di in range(window):
                for dj in range(window):
                    v = padded[i + di, j + dj]
                    diff = v - mean
                    var += diff * diff
            var /= count

            result[i, j] = np.sqrt(var)

    return result


def compute_roughness(elevation: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Roughness defined as local standard deviation of elevation
    in a sliding window.

    Returns RAW std-dev values, no normalization.

    Parameters
    ----------
    elevation : np.ndarray
        2D elevation array (H, W).
    window : int
        Window size for rolling std.

    Returns
    -------
    rough : np.ndarray
        (H, W) float32 array of local std.
    """
    elevation = elevation.astype(np.float32, copy=False)

    try:
        rough = rolling_std_numba(elevation, window)
    except Exception:
        from scipy.ndimage import generic_filter  # type: ignore

        rough = generic_filter(elevation, np.std, size=window)

    return rough.astype(np.float32)


# ------------------------------------------------------------
# Plan curvature
# ------------------------------------------------------------
def compute_plan_curvature(elevation: np.ndarray) -> np.ndarray:
    """
    Plan curvature derived from second derivatives of elevation.

    Returns RAW curvature values (can be positive or negative),
    no normalization.

    Parameters
    ----------
    elevation : np.ndarray
        2D elevation array (H, W).

    Returns
    -------
    curv : np.ndarray
        (H, W) float32 array of plan curvature.
    """
    elevation = elevation.astype(np.float32, copy=False)

    dy, dx = np.gradient(elevation)
    dxy, dxx = np.gradient(dx)
    dyy, dyx = np.gradient(dy)

    numerator = dxx * dy**2 - 2.0 * dxy * dx * dy + dyy * dx**2
    denominator = (dx**2 + dy**2 + 1e-8) ** 1.5 + 1e-8
    curv = numerator / denominator

    return curv.astype(np.float32)


# ------------------------------------------------------------
# Combined physics feature extractor
# ------------------------------------------------------------
def compute_phys_v4(elevation_full: np.ndarray, res=64, scale=1.0) -> np.ndarray:
    """
    Build a 4-channel physics tensor from elevation:

        [ flow_raw, tpi_raw, roughness_raw, plan_curvature_raw ]

    All channels are RAW (no 0-1 scaling, no mean/std).
    Any normalization should be applied later in the data
    preprocessing / dataloader pipeline.

    Parameters
    ----------
    elevation_full : np.ndarray
        (H, W, C) or (H, W) array; channel 0 is assumed to be elevation.
    res : int or "full"
        Sampling step for BFS sources in flow computation.
    scale : float
        Optional downscale factor for flow computation.

    Returns
    -------
    phys : np.ndarray
        (H, W, 4) array:
            0: flow accumulation
            1: TPI
            2: roughness
            3: plan curvature
    """
    if elevation_full.ndim == 3:
        elevation = elevation_full[:, :, 0].astype(np.float32)
    else:
        elevation = elevation_full.astype(np.float32)

    flow = compute_flow(elevation, res=res, scale=scale)
    tpi = compute_tpi(elevation, radius=5)
    rough = compute_roughness(elevation, window=3)
    curv = compute_plan_curvature(elevation)

    return np.stack(
        [
            flow.astype(np.float32),
            tpi.astype(np.float32),
            rough.astype(np.float32),
            curv.astype(np.float32),
        ],
        axis=-1,
    )


if __name__ == "__main__":
    # Minimal self-test (you can adapt paths as needed)
    import glacier_mapping.data.slice as fn  # adjust import if necessary

    dem = fn.read_tiff("/data/baryal/HKH/DEM/image1.tif")
    dem_np = np.transpose(dem.read(), (1, 2, 0)).astype(np.float32)
    dem_np = np.nan_to_num(dem_np)
    elevation = dem_np[:, :, 0][:, :, None]

    phys_output = compute_phys_v4(elevation, res=64, scale=0.3)
    print("Physics output shape:", phys_output.shape, "dtype:", phys_output.dtype)
