import abc


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError
