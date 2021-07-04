import onnxruntime
import cv2 
import numpy as np 

from . import gpu_nms

def load_onnx_model(onnx_path): 
    onnx_session = onnxruntime.InferenceSession(onnx_path)
    return onnx_session

def get_output_name(onnx_session):
    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)
    return output_name

def transform(image, target_shape=(960, 960)):
    image_height, image_width, _ = image.shape
    ratio_h = target_shape[1] * 1.0 / image_height
    ratio_w = target_shape[0] * 1.0 / image_width
    image = cv2.resize(image, target_shape)
    return image, ratio_h, ratio_w

def onnx_inference(onnx_session, num_classes, image, topk_candidates=1000): 

    output_name = get_output_name(onnx_session)

    image, ratio_h, ratio_w = transform(image)
    image = image.astype(np.float32)
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)

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

    boxes[:, 0] /= ratio_w
    boxes[:, 1] /= ratio_h
    boxes[:, 2] /= ratio_w
    boxes[:, 3] /= ratio_h

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
