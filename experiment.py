from prototype_dev import Env


class Experiment:
    def __init__(self, name, env):
        self.name = name
        self.env: Env = env
