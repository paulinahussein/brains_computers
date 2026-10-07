import os
import sys
import subprocess


# --------------------------------------------------
# argument
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python run_encoding_pieman.py sub-001")
    sys.exit(1)

subject = sys.argv[1]


# --------------------------------------------------
# paths
# --------------------------------------------------

BASE = "/nethome/phussein/brains/bert-brains"

FEATURE_DIR = os.path.join(
    BASE,
    "text_processing_and_transformers"
)

DATA_DIR = "/nethome/phussein/brains/bert-brains/pieman_data/"
assert os.path.exists(DATA_DIR + subject + ".nii.gz"), \
    f"Missing fMRI file: {DATA_DIR}{subject}.nii.gz"

RESULT_DIR = os.path.join(
    BASE,
    "results"
)

SCRIPT = os.path.join(
    BASE,
    "analysis",
    "banded_ridge_regression.py"
)


# --------------------------------------------------
# BERT representations
# --------------------------------------------------

representations = []

for layer in range(12):

    representations.append(
        (
            f"layer_{layer}_activations",
            f"pieman_bert-base-uncased_layer_{layer}_activations.npy"
        )
    )

    representations.append(
        (
            f"layer_{layer}_z_representations",
            f"pieman_bert-base-uncased_layer_{layer}_z_representations.npy"
        )
    )


# --------------------------------------------------
# run all encodings
# --------------------------------------------------

for name, filename in representations:

    feature_file = os.path.join(
        FEATURE_DIR,
        filename
    )

    save_dir = os.path.join(
        RESULT_DIR,
        f"pieman_encoding_{name}"
    )

    os.makedirs(save_dir, exist_ok=True)


    cmd = [
        "python",
        SCRIPT,
        subject,
        feature_file,
        name,
        DATA_DIR,
        "15",
        "277",
        save_dir
    ]


    print("\nRunning:")
    print(" ".join(cmd))


    subprocess.run(
        cmd,
        check=True
    )


print("\nFinished:", subject)