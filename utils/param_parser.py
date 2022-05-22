import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run HARNN.")

    # Data Parameters
    parser.add_argument("--train-file", nargs="?", default="../data/Train_sample.json", help="Training data.")
    parser.add_argument("--validation-file", nargs="?", default="../data/Validation_sample.json", help="Validation data.")
    parser.add_argument("--test-file", nargs="?", default="../data/Test_sample.json", help="Testing data.")
    parser.add_argument("--metadata-file", nargs="?", default="../data/metadata.tsv",
                        help="Metadata file for embedding visualization.")
    parser.add_argument("--word2vec-file", nargs="?", default="../data/word2vec_100.kv",
                        help="Word2vec file for embedding characters (the dim need to be the same as embedding dim).")

    # Model Hyperparameters
    parser.add_argument("--pad-seq-len", type=int, default=150, help="Padding sequence length. (depends on the data)")
    parser.add_argument("--embedding-type", type=int, default=1, help="The embedding type.")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Dimensionality of character embedding.")
    parser.add_argument("--lstm-dim", type=int, default=256, help="Dimensionality of LSTM neurons.")
    parser.add_argument("--lstm-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--attention-dim", type=int, default=200, help="Dimensionality of Attention neurons.")
    parser.add_argument("--attention-penalization", type=bool, default=True, help="Use attention penalization or not.")
    parser.add_argument("--fc-dim", type=int, default=512, help="Dimensionality for FC neurons.")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout keep probability.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of global part in scores cal.")
    parser.add_argument("--num-classes-list", type=list, default=[22, 298, 1475, 1],
                        help="Each number of labels in hierarchical structure. (depends on the task)")
    parser.add_argument("--total-classes", type=int, default=1796, help="Total number of labels. (depends on the task)")
    parser.add_argument("--topK", type=int, default=5, help="Number of top K prediction classes.")
    parser.add_argument("--threshold", type=float, default=0.24, help="Threshold for prediction classes.")

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch Size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--decay-rate", type=float, default=0.95, help="Rate of decay for learning rate.")
    parser.add_argument("--decay-steps", type=int, default=500, help="How many steps before decay learning rate.")
    parser.add_argument("--evaluate-steps", type=int, default=10, help="Evaluate model on val set after how many steps.")
    parser.add_argument("--norm-ratio", type=float, default=1.25,
                        help="The ratio of the sum of gradients norms of trainable variable.")
    parser.add_argument("--l2-lambda", type=float, default=0.0, help="L2 regularization lambda.")
    parser.add_argument("--checkpoint-steps", type=int, default=10, help="Save model after how many steps.")
    parser.add_argument("--num-checkpoints", type=int, default=5, help="Number of checkpoints to store.")

    # Misc Parameters
    parser.add_argument("--allow-soft-placement", type=bool, default=True, help="Allow device soft device placement.")
    parser.add_argument("--log-device-placement", type=bool, default=False, help="Log placement of ops on devices.")
    parser.add_argument("--gpu-options-allow-growth", type=bool, default=True, help="Allow gpu options growth.")

    return parser.parse_args()