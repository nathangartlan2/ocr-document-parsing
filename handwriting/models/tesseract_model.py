import pytesseract as pt
import models.model_base as base


class tesseract_model(base.model_base):

    def __init__(self) -> None:
        super().__init__()
        self.name = "tesseract"

    def train(self):
        pass

    def plot_training(self):
        pass

    def runTestInference(self,
                         images,
                         labels):

        print("Running inference on " + str(len(images)) + " images.")
        translations = []
        for img in images:
            translations.append(pt.image_to_string(image=img))

        print("Inference complete")
        return labels, translations

    def getName(self):
        return self.name
