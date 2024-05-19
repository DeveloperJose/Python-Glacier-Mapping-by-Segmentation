import multiprocessing.pool as mpp
import os
import shutil


def get_physics_from_run_name(run_name):
    if "phys" not in run_name:
        raise Exception("")
    start_idx = run_name.index("phys")
    end_idx = run_name.index("_", start_idx)

    # phys##
    phys_res_str = run_name[start_idx:end_idx]

    start_idx = end_idx + 1
    try:
        end_idx = run_name.index("_", end_idx + 1)
    except ValueError:
        end_idx = len(run_name)

    # s###
    phys_scale_str = run_name[start_idx:end_idx]

    physics_res = int(phys_res_str[4:])
    physics_scale = float(phys_scale_str[1:])
    print(phys_res_str, phys_scale_str, physics_res, physics_scale)
    return physics_res, physics_scale


def remove_and_create(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)


# https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
# istarmap.py for Python 3.8+


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
