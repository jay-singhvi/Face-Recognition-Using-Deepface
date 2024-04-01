# import matplotlib.pyplot as plt
import sys
import shutil
import os

os.environ["DEEPFACE_HOME"] = os.getcwd()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from deepface import DeepFace


class Face_Detector:
    def __init__(self, source_dir, destination_dir, print_debug=False) -> None:
        # Directories
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.print_debug = print_debug
        if self.print_debug:
            print("Source Dir: ", self.source_dir)
            print("Destination Dir: ", self.destination_dir)
        self.list_image_files()

    def list_image_files(self):
        """
        List all images in the source directory
        """
        if not os.path.exists(self.destination_dir):
            os.makedirs(self.destination_dir)
        # List all images in the source directory
        self.image_paths = [
            self.source_dir + f
            for f in os.listdir(self.source_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        if self.print_debug:
            print("Number of images: ", len(self.image_paths))
            print(self.image_paths)

    def move_image(self, image_path):
        """
        Move image to destination if faces were detected by all combinations.
        """
        destination_path = self.destination_dir + os.path.basename(image_path)
        shutil.move(image_path, destination_path)
        print(f"Moved: {image_path} -> {destination_path}")

    def udf_deepface_face_detector(self, image_path):
        """
        Detect faces using DeepFace
        """
        if self.print_debug:
            print("Start - udf_deepface_face_detector")
        detectors = [
            "opencv",
            "retinaface",
            "mtcnn",
            "ssd",
            "dlib",
            "mediapipe",
            "yolov8",
        ]
        counter = 0
        for detector in detectors:
            if self.print_debug:
                print(f"Start - {detector}")
            detected_aligned_face = DeepFace.extract_faces(
                img_path=image_path, detector_backend=detector, enforce_detection=False
            )

            if self.print_debug:
                print(
                    f"End - {detector} |  No. of Faces : {len(detected_aligned_face)}"
                )
                for _, i in enumerate(detected_aligned_face):
                    print(f" | confidence : {i['confidence']}")

            for _, i in enumerate(detected_aligned_face):
                # if i["confidence"] >= 0.75:
                plt.imshow(i["face"])
                plt.show()
                # counter += 1
                # break

            if counter >= 3:
                self.move_image(image_path)
                break

    def detect_faces(self):
        """
        Detect faces using DeepFace
        """
        for i in self.image_paths:
            if self.print_debug:
                print(f"Image: {i}")
            self.udf_deepface_face_detector(i)


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        file_name = os.path.basename(__file__)
        print(
            f"Usage: python {file_name} source_dir(path) destination_dir(path) print_debug(bool)"
        )
        print(
            f"Example: python {file_name} ./image_files/ ./image_files/with_faces/ True"
        )
        sys.exit(0)
    else:
        source_dir, destination_dir = sys.argv[1], sys.argv[2]
        print_debug = True if len(sys.argv) == 4 and sys.argv[3] == "True" else False
        obj1 = Face_Detector(source_dir, destination_dir, print_debug)
        obj1.detect_faces()
