import jittor as jt
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

jt.no_grad()
def get_bbox_from_mask(mask, enlarge: int=10):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    h, w = mask.shape
    # set zero-value for first raw, col and last row, col
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[h-1, :] = 0
    mask[:, w-1] = 0
    ## =====
    nonzero_idx =  jt.nonzero(mask)
    bbox_top = jt.min(nonzero_idx[:, 0])
    bbox_bottom = jt.max(nonzero_idx[:, 0])
    bbox_left = jt.min(nonzero_idx[:, 1])
    bbox_right = jt.max(nonzero_idx[:, 1])

    bbox_top = max(jt.to_int(bbox_top)-enlarge, 0)
    bbox_left = max(jt.to_int(bbox_left)-enlarge, 0)

    bbox_bottom = min(jt.to_int(bbox_bottom)+enlarge, h-1)
    bbox_right = min(jt.to_int(bbox_right)+enlarge, w-1)
    return bbox_top, bbox_left, bbox_bottom, bbox_right


if __name__ == '__main__':
    # test your mask file using this code
    test_dir = "/home/user/JNeuralTO/JNeRF/data/gummybear/mask"
    test_file = "0000.png"
    masks_np = cv2.imread(os.path.join(test_dir, test_file))
    masks_np = cv2.cvtColor(masks_np, cv2.COLOR_BGR2GRAY)
    masks_np = masks_np / 255.0
    masks_np = masks_np > 0.5
    masks = jt.Var(masks_np.astype(np.float32))
    aabb = get_bbox_from_mask(masks)
    bbox_top, bbox_left, bbox_bottom, bbox_right = aabb
    left = ([bbox_left, bbox_left], [bbox_top, bbox_bottom])
    right = ([bbox_right, bbox_right], [bbox_top, bbox_bottom])
    top = ([bbox_left, bbox_right], [bbox_top, bbox_top])
    bottom = ([bbox_left, bbox_right], [bbox_bottom, bbox_bottom])
    aabb_list = np.array(list(aabb))
    plt.imshow(masks.detach().numpy(), cmap='gray')
    # plot aabb
    plt.plot(left[0], left[1], color='r')
    plt.plot(right[0], right[1], color='r')
    plt.plot(top[0], top[1], color='r')
    plt.plot(bottom[0], bottom[1], color='r')
    plt.show()
