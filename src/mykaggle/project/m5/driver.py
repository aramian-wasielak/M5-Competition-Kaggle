import luigi

from mykaggle.project.m5.pipeline import RunPipeline

if __name__ == "__main__":
    luigi.build([RunPipeline(config_file="quick_one")], workers=1, local_scheduler=True)
