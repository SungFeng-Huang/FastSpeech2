import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, vctk


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "VCTK" in config["dataset"]:
        vctk.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if "LibriTTS" in config["dataset"] and "meta" in config:
        corpus_path = config["path"]["corpus_path"]
        raw_path = config["path"]["raw_path"]
        import os
        for dset in [config["meta"]["train"], config["meta"]["val"], config["meta"]["test"]]:
            config["path"]["corpus_path"] = os.path.join(corpus_path, dset)
            config["path"]["raw_path"] = os.path.join(raw_path, dset)
            main(config)
    else:
        main(config)
