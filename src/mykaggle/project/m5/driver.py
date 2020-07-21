import luigi

from mykaggle.project.m5.pipeline import RunPipeline

if __name__ == "__main__":
    luigi.build([RunPipeline()], workers=1, local_scheduler=True)
