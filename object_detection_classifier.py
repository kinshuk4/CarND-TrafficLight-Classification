import tensorflow as tf
import numpy as np
import os
from glob import glob
from object_detection_utilities import label_map_util
from object_detection_utilities import visualization_utils

NUM_DETECTIONS_ = 'num_detections:0'
DETECTION_CLASSES_ = 'detection_classes:0'
DETECTION_SCORES_ = 'detection_scores:0'
DETECTION_BOXES_ = 'detection_boxes:0'
IMAGE_TENSOR_ = 'image_tensor:0'
MIN_SCORE_THRESHOLD = .50

UNKNOWN = 'UNKNOWN'
YELLOW = 'Yellow'
GREEN = 'Green'
RED = 'Red'


class ObjectDetectionClassifier(object):
    def __init__(self, model_path, labels_path, num_classes, fx=None, fy=None):
        # set default value for no detection
        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            raise ValueError("Unable to find the model path or labels file.")

        self.label_map = label_map_util.load_labelmap(labels_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.image_np_deep = None
        self.detection_graph = self.get_tf_graph(model_path)
        self.fx = fx
        self.fy = fy

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(IMAGE_TENSOR_)

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(DETECTION_BOXES_)

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(DETECTION_SCORES_)
        self.detection_classes = self.detection_graph.get_tensor_by_name(DETECTION_CLASSES_)
        self.num_detections = self.detection_graph.get_tensor_by_name(NUM_DETECTIONS_)

        print("Loaded frozen model graph")

    def get_tf_graph(self, model_path):
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def detect_object(self, image):
        image_expanded = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        return boxes, scores, classes

    def classify_image(self, image):
        self.current_light = UNKNOWN
        (boxes, scores, classes) = self.detect_object(image)

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > MIN_SCORE_THRESHOLD:
                class_name = self.category_index[classes[i]]['name']
                if class_name == RED:
                    self.current_light = RED
                elif class_name == GREEN:
                    self.current_light = GREEN
                elif class_name == YELLOW:
                    self.current_light = YELLOW
                self.image_np_deep = image

        if self.fx and self.fy:
            perceived_width_x = (boxes[i][3] - boxes[i][1]) * 800
            perceived_width_y = (boxes[i][2] - boxes[i][0]) * 600
            perceived_depth_x = ((.1 * self.fx) / perceived_width_x)
            perceived_depth_y = ((.3 * self.fy) / perceived_width_y)

            estimated_distance = round((perceived_depth_x + perceived_depth_y) / 2)
            print("Distance (metres)", estimated_distance)

        return self.current_light

    @staticmethod
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def visualize_detection(self, image):
        (boxes, scores, classes) = self.detect_object(image)
        image_np = ObjectDetectionClassifier.load_image_into_numpy_array(image)
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np, boxes, classes, scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=6)
        return image_np
