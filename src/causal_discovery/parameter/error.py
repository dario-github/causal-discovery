class ScoreError(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)


class DataTypeError(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)