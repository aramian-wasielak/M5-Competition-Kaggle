import luigi

from mykaggle.project.m5.pipeline import * # TODO: change to RunPipeline

if __name__ == "__main__":
    #luigi.build([TrainModel(store_id="CA_1", pred_week=4)], workers=1, local_scheduler=True)

    luigi.build([RunPipeline()], workers=1, local_scheduler=True)
    #ProcessInputFiles().run()

# TODO:
#    - Files into classes
#    - Break model into train & sales
