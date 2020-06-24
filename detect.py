import cv2
import numpy as np
from ml_serving.drivers import driver


def detect_bboxes(drv: driver.ServingDriver, bgr_frame: np.ndarray,
                            threshold: float = 0.5, offset=(0, 0)):
    # Get boxes shaped [N, 5]:
    # xmin, ymin, xmax, ymax, confidence
    input_name, input_shape = list(drv.inputs.items())[0]
    output_name = list(drv.outputs)[0]
    inference_frame = cv2.resize(bgr_frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = drv.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    boxes[:, 0] = boxes[:, 0] * bgr_frame.shape[1] + offset[0]
    boxes[:, 2] = boxes[:, 2] * bgr_frame.shape[1] + offset[0]
    boxes[:, 1] = boxes[:, 1] * bgr_frame.shape[0] + offset[1]
    boxes[:, 3] = boxes[:, 3] * bgr_frame.shape[0] + offset[1]
    return boxes[:, :4], boxes[:, 4]
