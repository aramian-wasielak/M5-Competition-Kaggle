import logging
import os
import pickle

import luigi
import pandas as pd
import yaml

logger = logging.getLogger("luigi-interface")


class GlobalParams(luigi.Config):
    config_file = luigi.Parameter(default="")
    config = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config is None:
            # TODO: Config location decide
            base_yaml = yaml.full_load(open("mykaggle/project/m5/config/base.yaml"))
            config_yaml = yaml.full_load(
                open(f"mykaggle/project/m5/config/{self.config_file}.yaml")
            )
            config = {**base_yaml, **config_yaml}
            self.__class__.config = config
            yaml.dump(
                config, open(os.path.join(config["output_dir"], "params.yaml"), "w")
            )


class InputFile(luigi.LocalTarget):
    def __init__(self, file):
        config = GlobalParams().config
        self.path = os.path.join(config["input_dir"], file)
        super().__init__(self.path)

    def get_path(self):
        return self.path

    def load(self):
        path = self.get_path()
        logger.info(f"Loading file: {path}")
        return pd.read_csv(path)


class OutputFile(luigi.LocalTarget):
    def __init__(self, file):
        config = GlobalParams().config
        self.path = os.path.join(config["output_dir"], file)
        super().__init__(self.path)

    def get_path(self):
        return self.path

    def load(self):
        path = self.get_path()
        logger.info(f"Loading file: {path}")
        return pickle.load(open(path, "rb"))

    def save(self, df):
        path = self.get_path()
        logger.info(f"Saving file: {path}")
        pickle.dump(df, open(path, "wb"))


class InputTask(luigi.ExternalTask):
    input_file = luigi.Parameter()

    def output(self):
        return InputFile(self.input_file)
