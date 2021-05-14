import onnxruntime
import cv2 
import numpy as np 

from .utils import gpu_nms

def load_onnx_model(onnx_path): 
    onnx_session = onnxruntime.InferenceSession(onnx_path)
    return onnx_session

def get_output_name(onnx_session):
    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)
    return output_name

def transform(img, short_edge_size=800, max_size=1333, size_divisibility=32): 
    h, w = img.shape[:2]

    # resize
    size = short_edge_size
    scale = size * 1.0 / min(h, w)
    if h < w:
        new_h, new_w = size, scale * w
    else:
        new_h, new_w = scale * h, size
    if max(new_h, new_w) > max_size:
        scale = max_size * 1.0 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)

    assert img.shape[:2] == (h, w)

    ret = cv2.resize(img, (new_w, new_h))

    # padding
    image_height, image_width = ret.shape[:2]
    if size_divisibility > 0:
        stride = size_divisibility
        max_height = int(np.ceil(image_height / stride) * stride)
        max_width = int(np.ceil(image_width / stride) * stride)

    padding_size = (
        (0, max_height - image_height),
        (0, max_width - image_width),
        (0, 0),
    )
    padded = np.pad(ret, padding_size, constant_values=0.0)
    return padded, new_h * 1.0 / h


def onnx_inference(onnx_session, num_classes, image, topk_candidates=1000): 

    image, scale = transform(image)
    image = image.astype(np.float32)

    output_name = get_output_name(onnx_session)
    scores, boxes = onnx_session.run(
        output_name, input_feed={"input": image}
    )

    keep = scores.max(axis=1) > 0.1
    scores = scores[keep]
    boxes = boxes[keep]

    scores = scores.flatten()
    # Keep top k top scoring indices only.
    num_topk = min(topk_candidates, len(boxes))
    # torch.sort is actually faster than .topk (at least on GPUs)
    topk_idxs = np.argsort(scores)

    scores = scores[topk_idxs][-num_topk:]
    topk_idxs = topk_idxs[-num_topk:]

    # filter out the proposals with low confidence score
    shift_idxs = topk_idxs // num_classes
    classes = topk_idxs % num_classes
    boxes = boxes[shift_idxs]

    # out: [boxes, scores]
    boxes = boxes / scale

    return boxes, scores, classes

def run(onnx_session, image, class_names, score_thrs, nms_thr=0.6):
    num_classes = len(class_names)
    boxes, scores, cls_idxs = onnx_inference(onnx_session, num_classes, image)

    assert len(boxes) == len(scores) and len(boxes) == len(cls_idxs)

    if isinstance(score_thrs, float):
        keep = scores > max(score_thrs, 0.2)
    else:
        score_thrs = np.asarray(score_thrs)
        keep = scores > np.maximum(score_thrs[cls_idxs], 0.2)

    pred_boxes = np.concatenate(
        [boxes, scores[:, np.newaxis], cls_idxs[:, np.newaxis]], axis=1
    )
    pred_boxes = pred_boxes[keep]

    all_boxes = []
    for cls_idx in range(len(class_names)):
        keep_per_cls = pred_boxes[:, -1] == cls_idx
        if keep_per_cls.sum() > 0:
            pred_boxes_per_cls = pred_boxes[keep_per_cls].astype(np.float32)
            keep_idx = gpu_nms(pred_boxes_per_cls[:, :5], nms_thr)
            for idx in keep_idx:
                all_boxes.append(
                    {
                        "class_name": class_names[cls_idx],
                        "x": float(pred_boxes_per_cls[idx][0]),
                        "y": float(pred_boxes_per_cls[idx][1]),
                        "w": float(
                            pred_boxes_per_cls[idx][2] - pred_boxes_per_cls[idx][0]
                        ),
                        "h": float(
                            pred_boxes_per_cls[idx][3] - pred_boxes_per_cls[idx][1]
                        ),
                        "score": float(pred_boxes_per_cls[idx][4]),
                    }
                )
    return {"boxes": all_boxes}
