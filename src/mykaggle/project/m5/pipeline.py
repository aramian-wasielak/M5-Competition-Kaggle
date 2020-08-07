import logging
import os
import pickle
from datetime import datetime

import luigi
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

from mykaggle.common.utils import (
    convert_to_date,
    downcast_numeric_cols,
    label_encoder,
    time_features,
)

logger = logging.getLogger("luigi-interface")

NUM_DAYS_1_WEEK = 7


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


class ProcessInputFiles(luigi.Task):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            "sales": InputTask("sales_train_evaluation.csv"),
            "calendar": InputTask("calendar.csv"),
            "prices": InputTask("sell_prices.csv"),
        }

    def run(self):
        sales = self.input()["sales"].load()
        sales = sales[sales.store_id == self.store_id]

        calendar = self.input()["calendar"].load()
        prices = self.input()["prices"].load()

        sales = label_encoder(
            sales, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        )
        sales = downcast_numeric_cols(sales)

        # Switch from wide to long format
        id_columns = ["id", "item_id", "store_id", "dept_id", "cat_id", "state_id"]
        sales = sales.melt(id_vars=id_columns, var_name="d", value_name="units_sold")
        sales = downcast_numeric_cols(sales)

        calendar = label_encoder(
            calendar,
            ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"],
        )
        calendar = convert_to_date(calendar, ["date"])
        calendar = downcast_numeric_cols(calendar)

        prices = label_encoder(prices, ["item_id", "store_id"])
        prices = downcast_numeric_cols(prices)

        sales = sales.merge(calendar, how="left", on="d")

        # Extract the day number from the day id (eg., d_<day number> -> <day number>)
        sales["d"] = sales["d"].str.extract(r"(\d+)").astype(np.int64)
        sales = downcast_numeric_cols(sales)

        sales = sales.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

        # The merge above is causing some data type loss and so we need to re-label
        sales = label_encoder(sales, ["store_id", "item_id"])
        sales = downcast_numeric_cols(sales)

        sales = sales[sales.sell_price.notnull()]
        self.output().save(sales)

    def output(self):
        return OutputFile(f"features_sales_base_store_{self.store_id}.pickle")


class SalesTimeSeriesFeatures(luigi.Task):
    store_id = luigi.Parameter()

    def requires(self):
        return ProcessInputFiles(store_id=self.store_id)

    def run(self):
        sales = self.input().load()
        sales = sales.drop(
            columns=[
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
        )

        sales = time_features(sales, "date")
        sales = downcast_numeric_cols(sales)

        col = "units_sold"
        groupby = ["id"]
        lag_days = [1, 2, 7, 14, 28]

        for lag in lag_days:
            logger.info("lag: {}".format(lag))
            sales["{}_lag_{}".format(col, lag)] = sales.groupby(groupby)[col].transform(
                lambda x: x.shift(lag)
            )

        sales = downcast_numeric_cols(sales)

        col = "units_sold"
        groupby = ["id"]
        roll_days = [7, 14, 28]

        for roll in roll_days:
            logger.info("Rolling period: {}".format(roll))
            sales["{}_roll_mean_{}".format(col, roll)] = sales.groupby(groupby)[
                col
            ].transform(lambda x: x.shift(1).rolling(roll).mean())

        sales = downcast_numeric_cols(sales)
        self.output().save(sales)

    def output(self):
        return OutputFile(f"features_sales_ts_store_{self.store_id}.pickle")


class PrepareTrainData(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return {
            "sales_base": ProcessInputFiles(store_id=self.store_id),
            "sales_ts": SalesTimeSeriesFeatures(store_id=self.store_id),
        }

    def run(self):
        config = GlobalParams().config
        train_start_date = config["train_start_date"]
        train_end_date = config["train_end_date"]

        logger.info(
            "Preparing train data for range: {} - {}".format(
                train_start_date, train_end_date
            )
        )
        sales_base = self.input()["sales_base"].load()
        sales = sales_base[
            (train_start_date <= sales_base.date) & (sales_base.date <= train_end_date)
        ]
        del sales_base

        sales_features = self.input()["sales_ts"].load()
        sales_features = sales_features[
            (train_start_date <= sales_features.date)
            & (sales_features.date <= train_end_date)
        ]
        columns = ["id", "date"] + list(
            set(sales_features.columns) - set(sales.columns)
        )
        sales = sales.merge(
            sales_features[[c for c in sales_features.columns if c in columns]],
            how="left",
            on=["id", "date"],
        )
        del sales_features

        shift_days = self.pred_week * NUM_DAYS_1_WEEK
        sales[config["pred_target_col"]] = sales.groupby("id")[
            config["target_col"]
        ].shift(-shift_days)
        sales[config["pred_date_col"]] = sales.groupby("id")["date"].shift(-shift_days)
        sales = sales[sales[config["pred_date_col"]].notnull()]
        self.output().save(sales)

    def output(self):
        return OutputFile(f"train_store_{self.store_id}_week_{self.pred_week}.pickle")


class PrepareTestData(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return {
            "sales_base": ProcessInputFiles(store_id=self.store_id),
            "sales_ts": SalesTimeSeriesFeatures(store_id=self.store_id),
        }

    def run(self):
        config = GlobalParams().config
        test_start_date = config["test_start_date"]
        test_end_date = config["test_end_date"]

        logger.info(
            "Preparing test data for date range: {} - {}".format(
                test_start_date, test_end_date
            )
        )

        sales_base = self.input()["sales_base"].load()
        sales = sales_base[
            (test_start_date <= sales_base.date) & (sales_base.date <= test_end_date)
        ]
        del sales_base

        sales_features = self.input()["sales_ts"].load()
        sales_features = sales_features[
            (test_start_date <= sales_features.date)
            & (sales_features.date <= test_end_date)
        ]
        columns = ["id", "date"] + list(
            set(sales_features.columns) - set(sales.columns)
        )
        sales = sales.merge(
            sales_features[[c for c in sales_features.columns if c in columns]],
            how="left",
            on=["id", "date"],
        )
        del sales_features
        self.output().save(sales)

    def output(self):
        return OutputFile(f"test_store_{self.store_id}_week_{self.pred_week}.pickle")


class TrainModel(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return PrepareTrainData(store_id=self.store_id, pred_week=self.pred_week)

    def run(self):
        config = GlobalParams().config

        store_id = self.store_id
        logger.info(f"Training model for store_id: {store_id}")

        sales = self.input().load()
        train_dates = sorted(sales.date.unique())
        valid_start_date = min(train_dates[-config["valid_num_days"] :])

        X_train = sales[sales.date < valid_start_date].drop(
            [config["pred_target_col"], config["pred_date_col"], "date"], axis=1
        )
        y_train = sales[sales.date < valid_start_date][config["pred_target_col"]]

        X_valid = sales[sales.date >= valid_start_date].drop(
            [config["pred_target_col"], config["pred_date_col"], "date"], axis=1
        )
        y_valid = sales[sales.date >= valid_start_date][config["pred_target_col"]]

        print(GlobalParams().config["lgb_params"])
        model = LGBMRegressor(**GlobalParams().config["lgb_params"])
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric="rmse",
            early_stopping_rounds=10,
        )
        self.output().save(model)

    def output(self):
        return OutputFile(f"model_store_{self.store_id}_week_{self.pred_week}.pickle")


class RunPredictionStoreWeek(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return {
            "model": TrainModel(store_id=self.store_id, pred_week=self.pred_week),
            "test_data": PrepareTestData(
                store_id=self.store_id, pred_week=self.pred_week
            ),
        }

    def run(self):
        config = GlobalParams().config
        model = self.input()["model"].load()
        X_test = self.input()["test_data"].load()
        preds = X_test[["id", "date"]].copy()
        preds[config["pred_date_col"]] = preds["date"] + np.timedelta64(
            self.pred_week * NUM_DAYS_1_WEEK, "D"
        )
        X_test = X_test.drop("date", axis=1)

        preds[config["pred_target_col"]] = model.predict(X_test)
        self.output().save(preds)

        # Clean-up files that are not needed anymore
        file = (
            PrepareTrainData(store_id=self.store_id, pred_week=self.pred_week)
            .output()
            .path
        )
        if os.path.exists(file):
            os.remove(file)

    def output(self):
        return OutputFile(f"pred_store_{self.store_id}_week_{self.pred_week}.pickle")


class RunPredictionStore(luigi.WrapperTask):
    store_id = luigi.Parameter()

    def requires(self):
        return {
            w: RunPredictionStoreWeek(store_id=self.store_id, pred_week=w)
            for w in GlobalParams().config["pred_week_list"]
        }

    def run(self):
        # Clean-up files that are large but not needed at this stage
        file_list = [
            ProcessInputFiles(store_id=self.store_id).output().path,
            SalesTimeSeriesFeatures(store_id=self.store_id).output().path,
        ]
        for file in file_list:
            if os.path.exists(file):
                os.remove(file)


class RunPredictionAll(luigi.Task):
    def requires(self):
        return {
            s: RunPredictionStore(store_id=s)
            for s in GlobalParams().config["store_list"]
        }

    def run(self):
        config = GlobalParams().config
        preds = []
        for store_id in config["store_list"]:
            for pred_week in config["pred_week_list"]:
                preds.append(
                    RunPredictionStoreWeek(store_id=store_id, pred_week=pred_week)
                    .output()
                    .load()
                )

        preds_all = pd.concat(preds)
        preds_all = preds_all.sort_values(
            ["id", config["pred_date_col"]], ignore_index=True
        )
        self.output().save(preds_all)

    def output(self):
        return OutputFile("predictions.pickle")


class RunSubmission(luigi.Task):
    def requires(self):
        return {
            "predictions": RunPredictionAll(),
            "submission": InputTask(input_file="sample_submission.csv"),
        }

    def run(self):
        preds = self.input()["predictions"].load()
        submission = self.input()["submission"].load()

        preds["pred_num_day"] = np.int64(
            (preds["pred_date"] - min(preds["pred_date"]) + np.timedelta64(1, "D"))
            / np.timedelta64(1, "D")
        )
        preds["pred_day_id"] = preds["pred_num_day"].apply(lambda x: "F" + str(x))
        preds_pivot = preds[["id", "pred_units_sold", "pred_day_id"]].pivot_table(
            values="pred_units_sold", index="id", columns="pred_day_id"
        )
        submission = submission.set_index("id")
        submission.update(preds_pivot)
        submission.to_csv(self.output().path)

    def output(self):
        return OutputFile("submission.csv")


class RunPipeline(luigi.WrapperTask):
    config_file = luigi.Parameter()
    start_time = datetime.now()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GlobalParams(config_file=self.config_file)

    def requires(self):
        return RunSubmission()

    def run(self):
        logger.info(
            "Running pipeline is done: {}".format(datetime.now() - self.start_time)
        )
