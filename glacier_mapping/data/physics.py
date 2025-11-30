import numba
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# Static neighbor offsets (Numba-optimized)
# ------------------------------------------------------------
neighbor_offsets = np.array(
    [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0,  1),
        ( 1,  1),
        ( 1,  0),
        ( 1, -1),
        ( 0, -1),
    ],
    dtype=np.int32,
)

# ------------------------------------------------------------
# Optimized BFS (identical behavior, much faster)
# ------------------------------------------------------------
@numba.njit()
def bfs_fast(im, water_allpath, source, queue_r, queue_c, visited):
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

        # Original accumulation rule: every visited cell except source gets +1
        if not (ur == sr and uc == sc):
            water_allpath[ur, uc] += 1.0

        # Loop over 8 neighbors using static offsets
        for k in range(8):
            vr = ur + neighbor_offsets[k, 0]
            vc = uc + neighbor_offsets[k, 1]

            # Bounds check
            if vr < 0 or vr >= rows or vc < 0 or vc >= cols:
                continue

            # Downhill rule (unchanged)
            if not visited[vr, vc] and im[vr, vc] < curr_elev:
                visited[vr, vc] = True
                queue_r[queue_end] = vr
                queue_c[queue_end] = vc
                queue_end += 1

    return water_allpath

# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------
@numba.njit()
def mean_std(im):
    return (im - im.mean()) / im.std()

@numba.njit()
def min_max(im):
    return (im - im.min()) / (im.max() - im.min() + 1e-8)

# ------------------------------------------------------------
# Resize helper with cv2 / skimage / PIL fallbacks
# ------------------------------------------------------------
def resize(arr, new_rows, new_cols):
    try:
        import cv2
        arr_norm = min_max(arr)
        arr_uint8 = (arr_norm * 255).astype(np.uint8)
        resized = cv2.resize(arr_uint8, (new_cols, new_rows), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32) / 255.0

    except ImportError:
        try:
            from skimage.transform import resize as sk_resize
            arr_norm = min_max(arr)
            return sk_resize(arr_norm, (new_rows, new_cols), order=1, preserve_range=True).astype(np.float32)

        except ImportError:
            arr = min_max(arr) * 255
            arr_uint8 = arr.astype(np.uint8)
            img = Image.fromarray(arr_uint8)
            resized = np.array(img.resize((new_cols, new_rows)))
            return resized.astype(np.float32) / 255.0

# ------------------------------------------------------------
# Main phys_v2 computation (with fast BFS)
# ------------------------------------------------------------
def compute_phys_v2(elevation, res=64, scale=0.3):
    # Extract elevation 2D
    if len(elevation.shape) == 3:
        elevation_2d = elevation[:, :, 0]
    else:
        elevation_2d = elevation

    original_shape = elevation_2d.shape

    # Downscale if needed
    if scale != 1:
        new_rows = int(original_shape[0] * scale)
        new_cols = int(original_shape[1] * scale)
        elevation_2d = resize(elevation_2d, new_rows, new_cols)

    rows, cols = elevation_2d.shape

    # Water accumulation map
    water_allpath = np.zeros((rows, cols), dtype=np.float32)

    # Allocate reusable BFS buffers ONCE
    queue_r = np.zeros(rows * cols, dtype=np.int32)
    queue_c = np.zeros(rows * cols, dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=np.bool_)

    # Perform BFS from sampled grid points
    if res == "full":
        res = 1

    for i in range(0, rows, res):
        for j in range(0, cols, res):
            bfs_fast(elevation_2d, water_allpath, (i, j), queue_r, queue_c, visited)

    # Upscale back to original resolution
    if scale != 1:
        water_allpath = resize(water_allpath, original_shape[0], original_shape[1])

    # Normalize output and return as [H, W, 1]
    water_allpath = min_max(water_allpath)
    return water_allpath.reshape(original_shape[0], original_shape[1], 1)

# ------------------------------------------------------------
# Additional terrain features
# ------------------------------------------------------------
def compute_slope_magnitude(dem_np):
    slope_sin = dem_np[:, :, 1]
    slope_rad = np.arcsin(np.clip(slope_sin, -1, 1))
    slope_deg = slope_rad * 180.0 / np.pi
    return np.clip(slope_deg / 90.0, 0, 1)

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

def compute_tpi(elevation, radius=5):
    try:
        mean_elev = uniform_filter_numba(elevation, radius)
    except Exception:
        from scipy.ndimage import uniform_filter
        mean_elev = uniform_filter(elevation, size=radius)

    tpi = elevation - mean_elev
    tpi_max = np.abs(tpi).max() + 1e-8
    return (tpi / tpi_max + 1) / 2

@numba.njit()
def rolling_std_numba(arr, window):
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    half = window // 2

    padded = np.zeros((rows + 2 * half, cols + 2 * half))
    padded[half:half+rows, half:half+cols] = arr

    for i in range(rows):
        for j in range(cols):
            vals = padded[i:i+window, j:j+window].flatten()
            mean = 0.0
            for v in vals:
                mean += v
            mean /= vals.size

            var = 0.0
            for v in vals:
                diff = v - mean
                var += diff * diff
            var /= vals.size

            result[i, j] = np.sqrt(var)
    return result

def compute_roughness(elevation, window=3):
    try:
        rough = rolling_std_numba(elevation, window)
    except Exception:
        from scipy.ndimage import generic_filter
        rough = generic_filter(elevation, np.std, size=window)

    return min_max(rough)

def compute_plan_curvature(elevation):
    dy, dx = np.gradient(elevation)
    dxy, dxx = np.gradient(dx)
    dyy, dyx = np.gradient(dy)

    numerator = dxx * dy**2 - 2*dxy * dx * dy + dyy * dx**2
    denominator = (dx**2 + dy**2 + 1e-8)**1.5 + 1e-8
    curv = numerator / denominator

    curv_norm = curv / (np.abs(curv).max() + 1e-8)
    return (curv_norm + 1) / 2

# ------------------------------------------------------------
# Combined physics feature extractor
# ------------------------------------------------------------
def compute_phys_v3(elevation_full, dem_np, res=64, scale=1.0):
    elevation = elevation_full[:, :, 0]

    flow = compute_phys_v2(elevation_full, res, scale)[:, :, 0]
    slope = compute_slope_magnitude(dem_np)
    tpi = compute_tpi(elevation, radius=5)
    roughness = compute_roughness(elevation, window=3)
    curvature = compute_plan_curvature(elevation)

    return np.stack([flow, slope, tpi, roughness, curvature], axis=-1)

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import slice as fn

    dem = fn.read_tiff("/data/baryal/HKH/DEM/image1.tif")
    dem_np = np.transpose(dem.read(), (1, 2, 0)).astype(np.float32)
    dem_np = np.nan_to_num(dem_np)
    elevation = dem_np[:, :, 0][:, :, None]

    phys_output = compute_phys_v2(elevation, res=64)
    print(phys_output.shape)

    import matplotlib.pyplot as plt
    plt.imshow(phys_output[:, :, 0], cmap="gray")
    plt.savefig("physics_example.png")

