# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
from tflite_runtime import interpreter as tflite
import yaml
from utils import Colors

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
# from ultralytics.utils.plotting import Colors

# Declare as global variables, can be updated based trained model image size
img_width = 640
img_height = 640


class LetterBox:
    def __init__(
        self, new_shape=(img_width, img_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left


    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""

        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""

        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class Yolov8TFLite:
    def __init__(self, tflite_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8TFLite class.

        Args:
            tflite_model: Path to the TFLite model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """

        self.tflite_model = tflite_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        #self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]
        yamlPath = "coco8.yaml"
        with open(yamlPath, 'r', encoding='utf-8') as f:
            self.classes = yaml.load(f,Loader=yaml.SafeLoader)["names"]

        # # Generate a color palette for the classes
        # self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # Create color palette
        self.color_palette = Colors()

        self.pad_h = None
        self.pad_w = None

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)

        #print("image before", self.img)
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        letterbox = LetterBox(new_shape=[img_width, img_height], auto=False, stride=32)
        image = letterbox(image=self.img)
        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(image)
        # n, h, w, c
        image = img.astype(np.float32)

        return image / 255

    def postprocess_detction(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        boxes = []
        scores = []
        class_ids = []
        for pred in output:
            pred = np.transpose(pred)
            for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            gain = min(img_width / self.img_width, img_height / self.img_height)
            pad = (
                round((img_width - self.img_width * gain) / 2 - 0.1),
                round((img_height - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]
            if score > 0.25:
                print(box, score, class_id)
                # Draw the detection on the input image
                self.draw_detections(input_image, box, score, class_id)

        return input_image

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    # 屏蔽 掉那些不在检测框内的部分
    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    #讲解可看 https://blog.csdn.net/qq_40672115/article/details/134277752
    # x[:, 6:]与mask相关 x[:, :4]与bbox相关
    # masks = self.process_mask(output_mask, x[:, 6:], x[:, :4], self.img.shape)
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w]. -> 32, 160, 160 分割头输出
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms. -> n, 32  检测头输出的 32 维向量，可以理解为 mask 的权重
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape. -> n, 4  检测框
            im0_shape (tuple): the size of the input image (h,w,c). > 640, 640   输入网络中的图像 shape

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        #矩阵相乘，将检测头的权重与分割头的基向量相乘
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """
        im = self.img
        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            print("cls_:", cls_)
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            cv2.imwrite("demo.jpg", im)

    #根据onnx的代码改写的，适用于seg模型的两个输出头
    # 输入是模型推理的结果，即8400个预测框 1,8400,116 [cx,cy,w,h,class*80,32]

    def postprocess(self, output_detection, output_mask):
        output_detection[:, [0, 2]] *= img_width
        output_detection[:, [1, 3]] *= img_height
        print(output_detection)
        # Perform post-processing on the outputs to obtain output image.
        # img_output_detection = self.postprocess(self.img, output)


        print("output", output_detection.shape)
        output_detection = output_detection.transpose(0, 2, 1)
        x = output_detection
        print("x2", x.shape)    #(1, 8400, 116)
        # Predictions filtering by conf-threshold
        # 取class*80的列，计算了每个检测结果的置信度中的最大值，保留置信度高于阈值的检测结果。
        x = x[np.amax(x[..., 4:-32], axis=-1) > self.confidence_thres]
        print("x3", x.shape)    #(, 116)
        #重新拼接[位置信息4列][置信度最大值][置信度最大值索引][最后32列]
        x = np.c_[x[..., :4], np.amax(x[..., 4:-32], axis=-1), np.argmax(x[..., 4:-32], axis=-1), x[..., -32:]]
        print("x4", x.shape)    #(, 38)
        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], self.confidence_thres, self.iou_thres)]
        print("x5", x.shape)    #(, 38)

        shape = self.img.shape[:2]  # original image shape
        new_shape = (640, 640)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # # Bounding boxes boundary clamp
            # x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            # x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            output_mask = np.squeeze(output_mask, axis=0)
            output_mask = output_mask.transpose(2, 0, 1)

            print("output_mask", output_mask.shape)
            print("x", x.shape)
            print("self.img.shape", self.img.shape)

            # Process masks
            # x[:, 6:]与mask相关 x[:, :4]与bbox相关
            masks = self.process_mask(output_mask, x[:, 6:], x[:, :4], self.img.shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks

            print("x[..., :6]", x.shape)

        else:
            return [], [], []



    def main(self):
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        # Create an interpreter for the TFLite model
        interpreter = tflite.Interpreter(model_path=self.tflite_model)
        self.model = interpreter
        interpreter.allocate_tensors()

        # Get the model inputs
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("output_details",output_details)

        # Store the shape of the input for later use
        input_shape = input_details[0]["shape"]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        # Preprocess the image data
        img_data = self.preprocess()
        img_data = img_data
        # img_data = img_data.cpu().numpy()
        # Set the input tensor to the interpreter
        # print(input_details[0]["index"])
        print("img_data",img_data.shape)
        img_data = img_data.transpose((0, 2, 3, 1))

        scale, zero_point = input_details[0]["quantization"]
        interpreter.set_tensor(input_details[0]["index"], img_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor from the interpreter
        output_detection = interpreter.get_tensor(output_details[0]["index"])
        scale, zero_point = output_details[0]["quantization"]
        #output = (output.astype(np.float32) - zero_point) * scale

        print("output_detection",output_detection.shape)

        output_mask = interpreter.get_tensor(output_details[1]["index"])

        return self.postprocess(output_detection,output_mask)





        #return segments, masks


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        #"--model", type=str, default="/home/velonica/code/yolov8/yolov8s_saved_model/yolov8s_float32.tflite", help="Input your TFLite model."
        "--model", type=str, default="C:\\GitHub\\ultralytics\\examples\\YOLOv8-OpenCV-int8-tflite-Python/yolov8n-seg_float32.tflite", help="Input your TFLite model."
    )
    parser.add_argument("--img", type=str, default="C:\\work\\datasets\\coco\\test-dev\\voc_jpg_seg\\000000000139.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    # Create an instance of the Yolov8TFLite class with the specified arguments
    detection = Yolov8TFLite(args.model, args.img, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    boxes, segments, _ = detection.main()

    # Draw bboxes and polygons
    detection.draw_and_visualize(boxes, segments, vis=False, save=True)

    # # Display the output image in a window
    # cv2.imshow("Output", output_image)

    # # Wait for a key press to exit
    # cv2.waitKey(0)
    #cv2.imwrite("output_image.png",output_image)