import inspect
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
RESULTS_DIR = os.path.join(BASE_DIR, "test_results")


class DATASET:

    observers = None

    def __init__(self):
        self.fixations = Path()
        self.raw_fixations = Path()
        self.binary_fixations = Path()
        self.generalized_fixations = Path()
        self.stimuli = Path()
        self.test_set = Path()

    def ensureconfig(self):
        assert str(self.fixations) != "."
        assert str(self.raw_fixations) != "."
        assert str(self.binary_fixations) != "."
        assert str(self.generalized_fixations) != "."
        assert str(self.stimuli) != "."
        assert str(self.test_set) != "."


    def create_test_set(self):
        raise NotImplementedError


class PSD(DATASET):

    def __init__(self):
        super().__init__()

        self.fixations = Path("data/psd/fixations")
        self.raw_fixations = Path("data/psd/raw")
        self.binary_fixations = Path("data/psd/binary")
        self.generalized_fixations = Path("data/psd/generalized")
        self.stimuli = Path("data/psd/generalized")
        self.test_set = Path("data/psd/test")

        self.ensureconfig()

    def create_test_set(self):

        pass
