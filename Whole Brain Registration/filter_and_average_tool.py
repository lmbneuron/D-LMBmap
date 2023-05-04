import os
from typing import List

import streamlit as st
import numpy as np
from glob import glob
from tqdm import tqdm

import torch

from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack


class MainSurface:
    def __init__(self):
        self.vol_dict = None
        self.average_path_list = []
        self.create_layout()

    def create_layout(self):
        st.title("Select and average")
        with st.sidebar:
            # path_regex = st.text_input("the regex of the path")
            # path_list = get_path_list(path_regex)

            root = st.text_input("the root of the data")
            if os.path.exists(root):
                path_list = [os.path.join(root, i, i+".tiff") for i in os.listdir(root) if os.path.exists(os.path.join(root, i, i+".tiff"))]
            else:
                path_list = []
            st.text("\n".join(path_list))

            if len(path_list) > 0:
                self.vol_dict = get_vol_dict(path_list)
                value = st.slider(
                    'Select a range of values',
                    0, len(self.vol_dict[[i for i in self.vol_dict.keys()][0]])-1, 100, 1)

        if self.vol_dict is not None:
            for i, (name, vol) in enumerate(self.vol_dict.items()):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(vol[value])
                    with col2:
                        if st.checkbox(name):
                            self.average_path_list.append(name)

        with st.sidebar:
            if st.button("average"):
                st.session_state["average_clicked"] = True
                vol = average_template(self.average_path_list)
                st.session_state["average_template"] = vol
                if "invert_transformed" in st.session_state:
                    st.session_state.pop("invert_transformed")

            if st.button("invert transform"):
                if "average_template" not in st.session_state:
                    st.text("You need an template")
                else:
                    st.session_state["invert_transformed"] = True
                    st.session_state["invert_template"] = inverse_vol(st.session_state["average_template"],
                                                                      "/".join(str.split(self.average_path_list[0], '/')[
                                                                               :-2]) + "/space.pkl")

        if st.session_state.get("average_clicked", False):
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state["average_template"][value])
                with col2:
                    st.text('template\naveraged by\n"' + '",\n"'.join(self.average_path_list) + '"')

        if st.session_state.get("invert_transformed", False):
            with col1:
                st.image(st.session_state["invert_template"][value])
            with col2:
                st.text("inverted template\naveraged by\n" + "\n".join(self.average_path_list))


@st.cache
def get_vol_by_path(path: str):
    vol = read_tiff_stack(path)
    return vol


@st.cache
def get_path_list(path_regex: str):
    return glob(path_regex)


@st.cache
def get_vol_dict(path_list: List[str]):
    vol_dict = {}
    for path in tqdm(path_list):
        vol_dict[path] = read_tiff_stack(path)
        vol_dict[path] = process_data(vol_dict[path])
        vol_dict[path] = np.transpose(vol_dict[path], (2, 0, 1, 3))
    return vol_dict


def average_template(path_list: List[str]):
    vol = None
    count = 0
    for path in path_list:
        if vol is None:
            vol = read_tiff_stack(path).astype(np.float64)
        else:
            vol += read_tiff_stack(path)
        count += 1
    vol /= count
    vol = vol.astype(np.uint16)
    vol = process_data(vol)
    return vol


def inverse_vol(vol, transform_path):
    import pickle
    from torch.nn.functional import grid_sample
    with open(transform_path, "rb") as f:
        transform = pickle.load(f)

    vol = np.transpose(vol, (3, 0, 1, 2))
    minn, maxn = np.min(vol), np.max(vol)
    vol = (vol-minn) / (maxn - minn)
    vol = torch.tensor(vol[np.newaxis, ...], dtype=torch.float32)
    vol = grid_sample(vol, transform, align_corners=True)
    vol = vol.numpy()
    vol = vol[0]
    vol = np.transpose(vol, (1, 2, 3, 0))
    vol = vol * (maxn - minn) + minn
    vol = vol.astype(np.uint8)
    return vol


def process_data(vol):
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * 255
    vol = vol.astype(np.uint8)
    vol = np.tile(vol[..., np.newaxis], (1, 1, 1, 3))
    return vol


if __name__ == "__main__":
    main_surface = MainSurface()
