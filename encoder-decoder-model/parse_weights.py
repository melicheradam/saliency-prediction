import argparse
import os
import tensorflow as tf
import config
import data
import shutil
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namespace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """

    if os.path.isfile(args.path):
        data_path = args.path
    else:
        data_path = os.path.join(args.path, "")

    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"
    output = args.output

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    if args.phase == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path,
        "output": output
    }

    return paths

def extract_weights(dataset, paths, device):

    iterator = data.get_dataset_iterator("test", dataset, paths["data"])
    next_element, init_op = iterator
    input_images, original_shape, file_path = next_element
    graph_def = tf.GraphDef()
    model_name = "model_%s_%s.pb" % (dataset, device)

    if os.path.isfile(paths["best"] + model_name):
        with tf.gfile.Open(paths["best"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())
    else:
        with tf.gfile.Open(paths["weights"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())

    [predicted_maps] = tf.import_graph_def(graph_def,
                                           input_map={"input": input_images},
                                           return_elements=["output:0"])

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True

    shutil.rmtree(paths["images"])
    os.makedirs(paths["images"])

    with tf.Session(config=conf) as sess:

        sess.run(init_op)

        #for v in tf.get_default_graph().as_graph_def().node:
        #    print(v.name)

        graph = tf.get_default_graph()

        var_name = "import/encoder-output-reduced"
        w2 = graph.get_tensor_by_name(var_name + ':0')
        shape = w2.get_shape()
        weights = sess.run(w2)
        w2_saved = sess.run(w2)  # print out tensor
        weights_flatten = weights.flatten()
        #print(len(weights_flatten))
        np.savetxt(os.path.join(paths["output"], 'weights.csv'), weights_flatten, delimiter=',')

def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    phases_list = ["train", "test"]

    datasets_list = ["salicon", "mit1003", "cat2000",
                     "dutomron", "pascals", "osie", "fiwi", "personalized"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                        help="sets the network phase (allowed: train or test)")

    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")

    parser.add_argument("-p", "--path", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    parser.add_argument("-o", "--output",
                        help="Output path to store the extracted tensor weights")

    args = parser.parse_args()

    paths = define_paths(current_path, args)

    print("Storing reduced weights from the last encoder layer...")
    extract_weights(args.data, paths, config.PARAMS["device"])
    print("Weights stored !")

if __name__ == "__main__":
    main()
