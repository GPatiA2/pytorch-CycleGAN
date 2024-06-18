from ultralytics import YOLO
import numpy as np
import cv2


class SlicingDetector:

    def __init__(self, opt):

        self.model = YOLO(opt.weights_path)
        self.crop_size = opt.det_crop_size
        self.crop_overlap = opt.det_crop_overlap
        self.confidence = opt.det_confidence
        self.iou = opt.det_iou
        self.dist_overlap = opt.det_dist_overlap

    def slice_image(self, image):

        img_w, img_h = image.shape[1], image.shape[0]

        step_w = int(self.crop_size * (1 - self.crop_overlap))
        step_h = int(self.crop_size * (1 - self.crop_overlap))

        step_w = max(1, step_w)
        step_h = max(1, step_h)

        slices = []
        coordinates_slices = []

        for y in range(0, img_h, step_h):
            for x in range(0, img_w, step_w):
                slices.append(image[y: y + self.crop_size,
                              x: x + self.crop_size, :])
                coordinates_slices.append([x, y])

        return slices, coordinates_slices

    def keep_overlapping_panels_X(self, masks):
        for points in masks[0]:
            x, _ = points
            if x > self.crop_overlap * self.crop_size - self.dist_overlap:
                return True
        return False

    def keep_overlapping_panels_Y(self, masks):

        for points in masks[0]:
            _, y = points
            if y > self.crop_overlap * self.crop_size - self.dist_overlap:
                return True
        return False

    def keep_panels_in_frame(self, mask, im):

        for points in mask[0]:
            x, y = points
            if x < self.dist_overlap or x > im.shape[1] - self.dist_overlap:

                return False
            elif y < self.dist_overlap or y > im.shape[0] - self.dist_overlap:
                return False
        return True

    def predict_img(self, image):

        slices, coordinates = self.slice_image(image)

        results = self.model.predict(source=slices,
                                     conf=self.confidence,
                                     iou=self.iou,
                                     imgsz=self.crop_size,
                                     augment=False,
                                     save=False,
                                     stream=True)

        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for res, image, coord_slice in zip(
            results, slices, coordinates
        ):
            if len(res) <= 0:
                continue

            masks = res.masks.xy

            for m in masks:
                m = np.int32([m])
                if not self.keep_panels_in_frame(m, image):
                    continue
                if not ((coord_slice[0] == 0) or self.keep_overlapping_panels_X(m)):
                    continue
                if not ((coord_slice[1] == 0) or self.keep_overlapping_panels_Y(m)):
                    continue

                m = m + [coord_slice]

                full_mask = cv2.polylines(full_mask, [m], True, 255, 2)

        return full_mask


if __name__ == "__main__":

    from argparse import Namespace
    opt = Namespace(
        weights_path="/home/cvar/copilot-global/panel_segmentation/CM+JRT_T/yolov8n_NoAug_Sliced_128-05292024-130513/weights/best-ir-yolov8n-128.pt",
        det_crop_size=128,
        det_crop_overlap=0.4,
        det_confidence=0.5,
        det_iou=0.7,
        det_dist_overlap=4,
        image_path="/home/cvar/copilot-global/panel_segmentation/datasets/CalaMocha_T-T-split-202405131252/DJI_20230625104151_0001_T.JPG"
    )

    model = SlicingDetector(opt)
    image = cv2.imread(opt.image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.merge((image, image, image))
    mask = model.predict_img(image)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
