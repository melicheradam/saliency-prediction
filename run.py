import os
from pathlib import Path

import docker
import click
import shutil

from config import DATASET, PSD, BASE_DIR, RESULTS_DIR, CAT2000

DOCKER_CLIENT = docker.from_env()
DOCKER_VOLUME = BASE_DIR + ":/labs"
DOCKER_ENV = "PYTHONPATH=/labs"
AVAILABLE_DATASETS = ["PSD", "CAT2000"]


@click.group()
def cli():
    pass


@cli.command()
@click.option("--observer", help="Observer name (if not provided all observers will be evaluated!)", default="")
@click.option("--load-name", help="Name to load a serialized model, can be empty for the default VGG16 weights", default="")
@click.option("--save-name", help="Name which will be used to serialize this model", default="")
@click.option("--model-type", type=click.Choice(["generalized", "personalized"]), required=True)
@click.argument("dataset", type=click.Choice(AVAILABLE_DATASETS + ["SALICON"]))
def train(observer: str, load_name: str, save_name: str, model_type: str, dataset: str):
    dataset_class = _get_dataset(dataset)
    command_args = " train -d {}"

    if dataset == "SALICON":
        command_args = command_args.format("salicon")
        _train_model(load_name, save_name, command_args)

    # train a specific observer or generalized
    elif observer != "" or model_type == "generalized":
        _prepare_training(model_type, observer, dataset_class)
        # naming is not optimal here, but model takes this as an "dataset" argument
        command_args = command_args.format("personalized")
        _train_model(load_name, save_name, command_args, observer)

    # train all observers
    elif model_type == "personalized":
        for observer in dataset_class.observers:
            _prepare_training(model_type, observer, dataset_class)
            command_args = command_args.format("personalized")
            _train_model(load_name, save_name, command_args, observer)

    else:
        AttributeError("Wrong combination of input attributes")


@cli.command()
@click.option("--load-name", help="Name of the model which will be tested", required=True)
@click.option("--observer", help="Observer name (if not provided all observers will be tested!)", default="")
@click.option("--discrepancy", help="Produce discrepancy maps", is_flag=True)
@click.option("--model-type", type=click.Choice(["generalized", "personalized"]), required=True)
@click.argument("dataset", type=click.Choice(AVAILABLE_DATASETS + ["SALICON"]))
def test(observer: str, load_name: str, discrepancy: bool, model_type: str, dataset: str):
    command_args = " test -d {} -p {}"
    dataset_class = _get_dataset(dataset)

    if dataset == "SALICON":
        command_args = command_args.format("salicon", dataset_class.test_set.as_posix())
    else:
        command_args = command_args.format("personalized", dataset_class.test_set.as_posix())

    # test a specific observer or generalized
    if observer != "" or model_type == "generalized":
        _test_model(load_name, command_args, observer)
        if discrepancy:
            _produce_discrepancy(load_name, dataset_class, observer)
    # train all observers
    elif model_type == "personalized":
        for observer in dataset_class.observers:
            _test_model(load_name, command_args, observer)
            if discrepancy:
                _produce_discrepancy(load_name, dataset_class, observer)
    else:
        AttributeError("Wrong combination of input attributes")


@cli.command()
@click.argument("dataset", type=click.Choice(AVAILABLE_DATASETS))
def make_test_set(dataset: str):
    dataset_class = _get_dataset(dataset)
    print("Creating test image set in path " + str(dataset_class.test_set))
    dataset_class.create_test_set()


@cli.command()
@click.option("--load-name", help="Name of the model which will be evaluated", required=True)
@click.option("--observer", help="Observer name (if not provided all observers will be evaluated!)", default="")
@click.option("--model-type", type=click.Choice(["generalized", "personalized"]), required=True)
@click.option("--infogain-name", help="Full name of the model (including observer) which will be used in the Infogain metric", default="")
@click.argument("dataset", type=click.Choice(AVAILABLE_DATASETS))
def evaluate(observer: str, load_name: str, model_type: str, infogain_name: str, dataset: str):
    print("Evaluating performance of the model...")
    dataset_class = _get_dataset(dataset)

    _load_name = load_name
    if observer != "":
        _load_name = load_name + "_" + observer
    infogain_path = Path(os.path.join(RESULTS_DIR, *[_load_name, "saliency"])) if infogain_name == "" \
                else Path(os.path.join(RESULTS_DIR, *[infogain_name, "saliency"]))

    command_args = " -gt {} -sal {} -pg {} -bin {} -output {}"

    # evaluate a specific observer or generalized
    if observer != "" or model_type == "generalized":
        command_args = command_args.format(
            dataset_class.fixations.joinpath(observer).as_posix(), Path(os.path.join(RESULTS_DIR, *[_load_name, "saliency"])).as_posix(),
            infogain_path.as_posix(), dataset_class.binary_fixations.joinpath(observer).as_posix(),
            Path(os.path.join(RESULTS_DIR, *[_load_name, "evaluation.json"])).as_posix()
        )
        _run_in_docker("python2", "python src/evaluate_results.py", command_args)
    # evaluate all observers
    elif model_type == "personalized":
        for observer in dataset_class.observers:
            _load_name = load_name + "_" + observer
            infogain_path = Path(os.path.join(RESULTS_DIR, *[_load_name, "saliency"])) if infogain_name == "" \
                else Path(os.path.join(RESULTS_DIR, *[infogain_name, "saliency"]))
            _command_args = command_args.format(
                dataset_class.fixations.joinpath(observer).as_posix(), Path(os.path.join(RESULTS_DIR, *[_load_name, "saliency"])).as_posix(),
                infogain_path.as_posix(), dataset_class.binary_fixations.joinpath(observer).as_posix(),
                Path(os.path.join(RESULTS_DIR, *[_load_name, "evaluation.json"])).as_posix()
            )
            print("Evaluating model {}...".format(_load_name))
            _run_in_docker("python2", "python src/evaluate_results.py", _command_args)
    else:
        AttributeError("Wrong combination of input attributes")


@cli.command()
@click.argument("dataset", type=click.Choice(AVAILABLE_DATASETS))
def preprocess_dataset(dataset):
    dataset_class = _get_dataset(dataset)
    print("Preparing dataset...")
    if isinstance(dataset_class, PSD):
        command_args = " -fix {} -raw {} -binary {} -generalized {}"\
            .format(dataset_class.fixations.as_posix(), dataset_class.raw_fixations.as_posix(),
                    dataset_class.binary_fixations.as_posix(), dataset_class.generalized_fixations.as_posix())

        _run_in_docker("python3", "python src/psd/initialize_dataset.py", command_args)

    elif isinstance(dataset_class, CAT2000):
        dataset_class.create_dir_structure()
        command_args = " -raw {} -out {}"\
            .format(dataset_class.raw_fixations.as_posix(), dataset_class.binary_fixations.as_posix())

        _run_in_docker("python3", "python src/cat2000/generate_maps.py", command_args)


@cli.command()
@click.argument("model-name")
def show_results(model_name):
    command_args = " -res {} -name {}".format(RESULTS_DIR, model_name)
    _run_in_docker("python3", "python src/show_training_results.py", command_args)


def _load_model(model_name):
    print("Loading model named " + model_name + "...")
    _run_in_docker("python3", "bash encoder-decoder-model/load_model.sh ", model_name)


def _save_model(model_name):
    print("Saving model named " + model_name + "...")
    _run_in_docker("python3", "bash encoder-decoder-model/serialize_current_model.sh ", model_name)


def _train_model(load_name, save_name, command_args, observer=""):
    if load_name != "":
        _load_model(load_name)

    _run_in_docker("python3-tensorflow", "python encoder-decoder-model/main.py", command_args)

    if save_name != "":
        _save_name = save_name
        if observer != "":
            _save_name += "_" + observer
        _save_model(_save_name)


def _test_model(load_name, command_args, observer=""):
    _load_name = load_name
    if observer != "":
        _load_name += "_" + observer
    _load_model(_load_name)
    print("Predicting maps by model {}...".format(_load_name))

    shutil.rmtree(os.path.join(RESULTS_DIR, *[_load_name, "saliency"]), ignore_errors=True)
    Path(os.path.join(RESULTS_DIR, *[_load_name, "saliency"])).mkdir(parents=True)

    _run_in_docker("python3-tensorflow", "python encoder-decoder-model/main.py", command_args)

    shutil.copytree(os.path.join(BASE_DIR, *["encoder-decoder-model", "results", "images"]),
                    os.path.join(RESULTS_DIR, *[_load_name, "saliency"]), dirs_exist_ok=True)


def _produce_discrepancy(load_name, dataset: DATASET, observer=""):
    _load_name = load_name
    if observer != "":
        _load_name += "_" + observer

    print("Producing discrepancy maps for model {}...".format(load_name))
    shutil.rmtree(os.path.join(RESULTS_DIR, *[_load_name, "discrepancy"]), ignore_errors=True)
    Path(os.path.join(RESULTS_DIR, *[_load_name, "discrepancy"])).mkdir(parents=True)
    command_args = " -gt {} -sal {} -orig {} -out {}".format(
        dataset.fixations.as_posix(), Path(os.path.join(RESULTS_DIR, *[_load_name, "saliency"])).as_posix(),
        dataset.test_set.as_posix(), Path(os.path.join(RESULTS_DIR, *[_load_name, "discrepancy"])).as_posix()
    )
    _run_in_docker("python3-tensorflow", "python src/differentiate_maps.py", command_args)


def _prepare_training(model_type, observer_name, dataset: DATASET):
    print("Preparing data for training...")
    try:
        shutil.rmtree(os.path.join(*["encoder-decoder-model", "data", "personalized", "stimuli"]))
        shutil.rmtree(os.path.join(*["encoder-decoder-model", "data", "personalized", "val-stimuli"]))

        shutil.rmtree(os.path.join(*["encoder-decoder-model", "data", "personalized", "saliency"]))
        shutil.rmtree(os.path.join(*["encoder-decoder-model", "data", "personalized", "val-saliency"]))
    except FileNotFoundError:
        pass

    if model_type == "generalized":
        shutil.copytree(os.path.join(dataset.stimuli), os.path.join(*["encoder-decoder-model", "data", "personalized", "stimuli"]), dirs_exist_ok=True)
        shutil.copytree(os.path.join(dataset.generalized_fixations), os.path.join(*["encoder-decoder-model", "data", "personalized", "saliency"]), dirs_exist_ok=True)

    elif model_type == "personalized":
        shutil.copytree(os.path.join(dataset.stimuli), os.path.join(*["encoder-decoder-model", "data", "personalized", "stimuli"]), dirs_exist_ok=True)
        shutil.copytree(os.path.join(dataset.fixations, observer_name), os.path.join(*["encoder-decoder-model", "data", "personalized", "saliency"]), dirs_exist_ok=True)


def _get_dataset(dataset: str) -> DATASET:
    if dataset == "PSD":
        return PSD()
    elif dataset == "CAT2000":
        return CAT2000()


def _run_in_docker(image: str, command: str, args: str):
    instance = DOCKER_CLIENT.containers.run(image,
                                 command=command + args,
                                 volumes=[DOCKER_VOLUME],
                                 environment=[DOCKER_ENV],
                                 remove=True,
                                 detach=True,
                                 runtime="nvidia",
                                 device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                                 )

    output = instance.attach(stdout=True, stream=True)

    for line in output:
        print(line.decode("utf-8"), end="", flush=True)


if __name__ == '__main__':
    cli()
