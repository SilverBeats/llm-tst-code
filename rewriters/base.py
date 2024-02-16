import abc


class BaseRewriter:
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def rewrite(self):
        raise NotImplementedError

    def _process_input_text(self, raw_text: str, label: int):
        pass
