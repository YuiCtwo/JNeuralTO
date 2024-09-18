import jittor as jt

def get_bbox_from_mask(mask, enlarge: int):
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
