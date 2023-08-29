# See https://github.com/pytorch/serve/blob/master/model-archiver/README.md#handler
import torch
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier


class Handler(ImageClassifier):
    def image_processing(self, image):
        return transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size=(28, 28))])(
            image
        )

    def postprocess(self, data):
        result = []
        for pred in data:
            index = pred.argmax()
            label = [3, 7][index]
            confidence = round(pred[index].item(), 2)
            result.append({"label": label, "confidence": confidence})
        return result
