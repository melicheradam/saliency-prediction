import inspect
import os
import random
import shutil
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
RESULTS_DIR = os.path.join("test-results")


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

    observers = ["Sub_1", "Sub_2", "Sub_3", "Sub_4", "Sub_5", "Sub_6", "Sub_7", "Sub_8", "Sub_9", "Sub_10",
                 "Sub_11", "Sub_12", "Sub_13", "Sub_14", "Sub_15", "Sub_16", "Sub_17", "Sub_18", "Sub_19", "Sub_20",
                 "Sub_21", "Sub_22", "Sub_23", "Sub_24", "Sub_25", "Sub_26", "Sub_27", "Sub_28", "Sub_29", "Sub_30"]

    def __init__(self):
        super().__init__()

        self.fixations = Path("data/psd/fixations")
        self.raw_fixations = Path("data/psd/raw")
        self.binary_fixations = Path("data/psd/binary")
        self.generalized_fixations = Path("data/psd/generalized")
        self.stimuli = Path("data/psd/images")
        self.test_set = Path("data/psd/test")

        self.ensureconfig()

    def create_test_set(self):
        shutil.rmtree(os.path.join(BASE_DIR, self.test_set), ignore_errors=True)
        self.test_set.mkdir(parents=True)

        image_set = list(os.listdir(os.path.join(BASE_DIR, self.stimuli)))
        random.shuffle(image_set)

        for idx, image in enumerate(image_set):
            if idx == 320:
                break

            shutil.copyfile(os.path.join(BASE_DIR, *[self.stimuli, image]),
                            os.path.join(BASE_DIR, *[self.test_set, image]))


class CAT2000(DATASET):

    def __init__(self):
        super().__init__()

        # path to directory which contains FIXATIONLOCS, FIXATIONMAPS and Stimuli directories
        self.train_set_path = Path("data/cat2000/trainSet")

        self.fixations = Path("data/cat2000/fixations")
        self.raw_fixations = Path("data/cat2000/raw")
        self.binary_fixations = Path("data/cat2000/binary")
        self.generalized_fixations = Path("data/cat2000/fixations")
        self.stimuli = Path("data/cat2000/images")
        # here we want to test on whole dataset
        self.test_set = Path("data/cat2000/images")

        self.ensureconfig()

    def create_dir_structure(self):

        self.raw_fixations.mkdir(parents=True, exist_ok=True)
        self.rename_files(self.train_set_path.joinpath("FIXATIONLOCS"), self.raw_fixations)
        self.fixations.mkdir(parents=True, exist_ok=True)
        self.rename_files(self.train_set_path.joinpath("FIXATIONMAPS"), self.fixations)
        self.stimuli.mkdir(parents=True, exist_ok=True)
        self.rename_files(self.train_set_path.joinpath("Stimuli"), self.stimuli)
        self.binary_fixations.mkdir(parents=True, exist_ok=True)

    def rename_files(self, dir: Path, out_dir: Path):

        for folder in os.listdir(dir):
            for file in os.listdir(dir.joinpath(folder)):
                if str(file) == "Output":
                    continue
                new_name = str(folder) + "_" + str(file)
                shutil.copy(
                    dir.joinpath(folder).joinpath(file),
                    out_dir.joinpath(new_name)
                )
