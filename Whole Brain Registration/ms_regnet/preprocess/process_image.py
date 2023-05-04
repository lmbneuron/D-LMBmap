import numpy as np
import os


def process_my_data(AVGT):
    AVGT = AVGT.astype(np.float)
    AVGT = (AVGT-np.min(AVGT)) / (np.max(AVGT)-np.min(AVGT)) * 255
    AVGT = AVGT.astype(np.uint8)
    return AVGT
