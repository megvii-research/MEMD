import json
import argparse

import cv2
from loguru import logger

from memd.onnx_model import load_onnx_model, run


def inference(detector_path, input_img, model_json, output_path):

    img = cv2.imread(input_img)

    with open(model_json) as file:
        model_info = json.load(file)

    class_names = model_info["CLASS_NAMES"]
    onnx_session = load_onnx_model(detector_path)

    score_thrs = model_info["SCORE_THRESH_TEST"]
    nms_thrs = model_info["NMS_THRESH_TEST"]

    score_thrs = max(score_thrs, 0.5)

    res = run(onnx_session, img, class_names, score_thrs, nms_thrs)

    for box in res["boxes"]:
        cv2.rectangle(
            img,
            (int(box["x"]), int(box["y"])),
            (int(box["x"] + box["w"]), int(box["y"] + box["h"])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            "{}:{:.2f}".format(box["class_name"], box["score"]),
            (int(box["x"] - 10), int(box["y"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    if not output_path.endswith((".jpg", ".png")):
        output_path = f"{output_path}.jpg"

    cv2.imwrite(output_path, img)
    logger.info(f"Inference on an image and write to {output_path}")

    box_num = len(res["boxes"])

    return box_num

def parse_args():
    parser = argparse.ArgumentParser("Electric Moped Detector")
    parser.add_argument(
        "--detector", default="./models/model.onnx", help="The path to onnx detector. "
    )
    parser.add_argument(
        "--input-img", default="./demo/sample.jpg", help="The path to demo image to inference. "
    )
    parser.add_argument(
        "--model-json", default="./memd/config/config.json", help="The input path of model info(json). "
    )
    parser.add_argument(
        "--output-path", default="./demo/output.jpg", help="The path of output images. "
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args() 
    print(args)
    inference(args.detector, args.input_img, args.model_json, args.output_path)


if __name__ == "__main__":
    main()
