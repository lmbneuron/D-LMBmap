import numpy as np
import cv2

from ms_regnet.tools.io import read_tiff_stack


def main():
    path_list = [
        r"C:\Users\haimiao\Desktop\210406_Adult Brain 23 in DBE_10-41-58_notransfer\fix_simi.tiff",
        r"C:\Users\haimiao\Desktop\210406_Adult Brain 23 in DBE_10-41-58_notransfer\reg_simi.tiff",

    ]

    tarpath = r"C:\Users\haimiao\Desktop\210406_Adult Brain 23 in DBE_10-41-58_notransfer\test.tiff",
    np_list = [read_tiff_stack(path) for path in path_list]

    newtiff = np.concatenate(np_list, axis=1)
    # newtiff = np.transpose(newtiff, (2, 0, 1))
    newtiff = newtiff.astype(np.uint8)
    # write_tiff_stack(newtiff, tarpath)
    export_as_video(newtiff, r"C:\Users\haimiao\Desktop\210406_Adult Brain 23 in DBE_10-41-58_notransfer\combine.avi")


def export_as_video(vol, path):
    vol = vol.astype(np.uint8)

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (vol.shape[2], vol.shape[1]))

    for img in vol:
        img = img[..., np.newaxis]
        img = np.tile(img, (1, 1, 3))
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()