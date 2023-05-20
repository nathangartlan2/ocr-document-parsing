from abc import ABC, abstractmethod


class model_base(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def plot_training(self):
        pass

    @abstractmethod
    def runTestInference(images, labels):
        pass

    @abstractmethod
    def getName(self):
        pass
