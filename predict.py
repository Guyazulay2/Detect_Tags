"""Sample prediction script for TensorFlow SavedModel"""
import argparse
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import cv2 as cv

MODEL_FILENAME = 'saved_model.pb'
LABELS_FILENAME = 'labels.txt'


class ObjectDetection:
    OUTPUT_TENSOR_NAMES = ['detected_boxes', 'detected_scores', 'detected_classes']

    def __init__(self, model_filename):
        model = tf.saved_model.load(os.path.dirname(model_filename))
        self.serve = model.signatures['serving_default']
        self.input_shape = self.serve.inputs[0].shape[1:3]

    def predict_image(self, image):
        dim = tuple(self.input_shape)
        image = cv.resize(image, dim)
        inputs = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        inputs = tf.convert_to_tensor(inputs)
        outputs = self.serve(inputs)
        return [outputs[n] for n in self.OUTPUT_TENSOR_NAMES]


def predict_video(model_filename):
    od_model = ObjectDetection(model_filename)
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        predictions = od_model.predict_image(frame)

        for bbox, conf, _ in zip(*predictions):
            if conf > 0.5:
                start = (bbox[0]*w, bbox[1]*h)
                end = (bbox[2]*w, bbox[3]*h)
                text_org = (bbox[0]*w, bbox[1]*h-5)
                color = (255,0,0)
                frame = cv.rectangle(frame, start, end, color, 2)
                frame = cv.putText(frame, f"tag-{np.round(conf.numpy(), 2)}", text_org, cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv.LINE_AA)
                
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser('Object Detection for Custom Vision TensorFlow model')
    parser.add_argument('--model_filename', type=str, default=MODEL_FILENAME, help='Filename for the tensorflow model')
    parser.add_argument('--labels_filename', type=str, default=LABELS_FILENAME, help='Filename for the labels file')
    args = parser.parse_args()

    predictions = predict_video(args.model_filename)

    with open(args.labels_filename) as f:
        labels = [l.strip() for l in f.readlines()]

if __name__ == '__main__':
    main()