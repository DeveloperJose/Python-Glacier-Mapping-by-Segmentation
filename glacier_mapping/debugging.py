import pathlib
import yaml

from addict import Dict
import streamlit as st
import numpy as np

from data.slice import get_tiff_np, read_shp, read_tiff, get_mask
from model.frame import Framework
import model.functions as fn

fname = pathlib.Path("image5.tif")
conf = Dict(yaml.safe_load(open('./conf/predict_slices.yaml')))
tiff_fname = pathlib.Path(conf["tiff_dir"]) / fname

labels_fname = pathlib.Path('/home/jperez/data/HKH_raw/labels/HKH_CIDC_5basins_all.shp')

@st.cache_data
def get_shp(fname):
    return read_shp(fname)

with st.status("Reading Shapefile"):
    labels = get_shp(labels_fname)

with st.status("Reading mask"):
    st.text(tiff_fname.exists())
    label_mask = get_mask(tiff_fname, labels)
    print(label_mask.shape)
    st.image(label_mask[:, :, 0])
    st.image(label_mask[:, :, 1])

runs_dir = pathlib.Path(conf.runs_dir)
output_dir = pathlib.Path(conf.output_dir)
checkpoint_path = runs_dir / conf.run_name / 'models' / 'model_best.pt'
frame: Framework = Framework.from_checkpoint(checkpoint_path, device=conf.gpu_rank, testing=True)

D_split = {}
pred_dir = pathlib.Path('../pred_runs/images/multi_phys64_s1/t=0.5')
all_pred_fnames = list(pred_dir.glob('*.tif'))
for pred_fname in all_pred_fnames:
    s = pred_fname.name.split('_')
    im_fname = s[0]
    split = s[1][:-4]
    D_split[im_fname] = split


split = D_split[fname.name[:-4]]
pred_fname =  pred_dir/ f'{fname.stem}_{split}.tif'
pred_tiff = np.transpose(read_tiff(pred_fname).read(), (1, 2, 0)).astype(np.uint8)
mask = np.sum(pred_tiff[:, :, :3], axis=2) < 0.01
y_true = frame.get_y_true(label_mask, mask)
print(y_true.shape)
st.image(y_true[:,:])
st.image(pred_tiff[:,:,0])
st.image(pred_tiff[:,:,1])

runs_dir = pathlib.Path(conf.runs_dir)
run_name: str = conf.run_name
physics_scale = conf.physics_scale
physics_res = conf.physics_res
gpu_rank: int = conf.gpu_rank
window_size = conf.window_size
threshold = conf.threshold
tiff_dir = pathlib.Path(conf.tiff_dir)
dem_dir = pathlib.Path(conf.dem_dir)
labels_path = pathlib.Path(conf.labels_path)

idx=0
dem_fname = dem_dir / fname
with st.status("Getting tiff"):
    x_arr = get_tiff_np(
        tiff_fname,
        dem_fname,
        physics_res=physics_res,
        physics_scale=physics_scale,
        verbose=(idx == 0),
    )
with st.status("Predicting whole"):
    y_pred, mask = frame.predict_whole(x_arr, window_size, threshold)
y_true = frame.get_y_true(label_mask, mask)
st.image(y_pred)
st.image(y_true)