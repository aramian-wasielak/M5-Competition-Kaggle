import os
import pickle
from datetime import datetime

import luigi
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import mykaggle.project.m5.config as config
from mykaggle.common.utils import (
    convert_to_date,
    downcast_numeric_cols,
    label_encoder,
    time_features,
)


class InputFile(luigi.LocalTarget):
    def __init__(self, file):
        super().__init__(str(config.input_dir / file))

    def get_path(self):
        return self.path

    def load(self):
        return pd.read_csv(self.get_path())


class OutputFile(luigi.LocalTarget):
    def __init__(self, file):
        super().__init__(str(config.output_dir / file))

    def get_path(self):
        return self.path

    def load(self):
        return pd.read_pickle(self.get_path())

    def save(self, df):
        df.to_pickle(self.get_path())


class CheckInputFiles(luigi.ExternalTask):
    def output(self):
        return [
            InputFile("sales_train_evaluation.csv"),
            InputFile("calendar.csv"),
            InputFile("sell_prices.csv"),
        ]


class ProcessInputFiles(luigi.Task):
    store_id = luigi.Parameter()

    def requires(self):
        return CheckInputFiles()

    def run(self):
        sales = InputFile("sales_train_evaluation.csv").load()
        sales = sales[sales.store_id == self.store_id]

        calendar = InputFile("calendar.csv").load()
        prices = InputFile("sell_prices.csv").load()

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

        OutputFile(f"features_sales_base_store_{self.store_id}.pickle").save(sales)

    def output(self):
        return OutputFile(f"features_sales_base_store_{self.store_id}.pickle")


class SalesTimeSeriesFeatures(luigi.Task):
    store_id = luigi.Parameter()

    def requires(self):
        return ProcessInputFiles(store_id=self.store_id)

    def run(self):
        sales = pd.read_pickle(
            config.output_dir / f"features_sales_base_store_{self.store_id}.pickle"
        )

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
            print("lag: {}".format(lag))
            sales["{}_lag_{}".format(col, lag)] = sales.groupby(groupby)[col].transform(
                lambda x: x.shift(lag)
            )

        sales = downcast_numeric_cols(sales)

        col = "units_sold"
        groupby = ["id"]
        roll_days = [7, 14, 28]

        for roll in roll_days:
            print("Rolling period: ", roll)
            sales["{}_roll_mean_{}".format(col, roll)] = sales.groupby(groupby)[
                col
            ].transform(lambda x: x.shift(1).rolling(roll).mean())

        sales = downcast_numeric_cols(sales)

        sales.to_pickle(
            config.output_dir / f"features_sales_ts_store_{self.store_id}.pickle"
        )

    def output(self):
        return luigi.LocalTarget(
            str(config.output_dir / f"features_sales_ts_store_{self.store_id}.pickle")
        )


class PrepareTrainData(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return [
            ProcessInputFiles(store_id=self.store_id),
            SalesTimeSeriesFeatures(store_id=self.store_id),
        ]

    def run(self):
        NUM_DAYS_1_WEEK = 7

        sales_base = pd.read_pickle(
            config.output_dir / f"features_sales_base_store_{self.store_id}.pickle"
        )

        train_start_date = config.train_start_date
        train_end_date = config.train_end_date

        print("Date range: {} - {}".format(train_start_date, train_end_date))

        sales_base = pd.read_pickle(
            config.output_dir / f"features_sales_base_store_{self.store_id}.pickle"
        )
        sales = sales_base[
            (train_start_date <= sales_base.date) & (sales_base.date <= train_end_date)
        ]
        del sales_base

        print("sales count: {}".format(len(sales)))

        sales_features = pd.read_pickle(
            config.output_dir / f"features_sales_ts_store_{self.store_id}.pickle"
        )

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
        sales[config.pred_target_col] = sales.groupby("id")[config.target_col].shift(
            -shift_days
        )
        sales[config.pred_date_col] = sales.groupby("id")["date"].shift(-shift_days)
        sales = sales[sales[config.pred_date_col].notnull()]

        sales.to_pickle(
            config.output_dir
            / f"train_store_{self.store_id}_week_{self.pred_week}.pickle"
        )

    def output(self):
        return luigi.LocalTarget(
            str(
                config.output_dir
                / f"train_store_{self.store_id}_week_{self.pred_week}.pickle"
            )
        )


class PrepareTestData(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return [
            ProcessInputFiles(store_id=self.store_id),
            SalesTimeSeriesFeatures(store_id=self.store_id),
        ]

    def run(self):
        sales_base = pd.read_pickle(
            config.output_dir / f"features_sales_base_store_{self.store_id}.pickle"
        )

        test_start_date = config.test_start_date
        test_end_date = config.test_end_date

        print("Test Date Range: {} - {}".format(test_start_date, test_end_date))

        sales_base = pd.read_pickle(
            config.output_dir / f"features_sales_base_store_{self.store_id}.pickle"
        )
        sales = sales_base[
            (test_start_date <= sales_base.date) & (sales_base.date <= test_end_date)
        ]
        del sales_base

        print("sales count: {}".format(len(sales)))

        sales_features = pd.read_pickle(
            config.output_dir / f"features_sales_ts_store_{self.store_id}.pickle"
        )

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

        sales.to_pickle(
            config.output_dir
            / f"test_store_{self.store_id}_week_{self.pred_week}.pickle"
        )

    def output(self):
        return luigi.LocalTarget(
            str(
                config.output_dir
                / f"test_store_{self.store_id}_week_{self.pred_week}.pickle"
            )
        )


class TrainModel(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return PrepareTrainData(store_id=self.store_id, pred_week=self.pred_week)

    def run(self):
        store_id = self.store_id
        print(f"store_id: {store_id}")

        sales = pd.read_pickle(
            config.output_dir
            / f"train_store_{self.store_id}_week_{self.pred_week}.pickle"
        )
        print(sales.head())

        train_dates = sorted(sales.date.unique())

        valid_start_date = min(train_dates[-20:])

        # TODO: config driven
        np.max(np.sort(sales.date)[:10])

        X_train = sales[sales.date < valid_start_date].drop(
            [config.pred_target_col, config.pred_date_col, "date"], axis=1
        )
        y_train = sales[sales.date < valid_start_date][config.pred_target_col]

        X_valid = sales[sales.date >= valid_start_date].drop(
            [config.pred_target_col, config.pred_date_col, "date"], axis=1
        )
        y_valid = sales[sales.date >= valid_start_date][config.pred_target_col]

        # TODO: Config file
        # Train and validate
        model = LGBMRegressor(
            n_estimators=2,  # 2000,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=8,
            num_leaves=50,
            min_child_weight=300,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric="rmse",
            verbose=20,
            early_stopping_rounds=10,
        )  # TODO: 50class TrainModel(luigi.Task):

        pickle.dump(
            model,
            open(
                config.output_dir
                / f"model_store_{self.store_id}_week_{self.pred_week}.pickle",
                "wb",
            ),
        )

    def output(self):
        return luigi.LocalTarget(
            str(
                config.output_dir
                / f"model_store_{self.store_id}_week_{self.pred_week}.pickle"
            )
        )


class RunPredictionStoreWeek(luigi.Task):
    store_id = luigi.Parameter()
    pred_week = luigi.IntParameter()

    def requires(self):
        return [
            TrainModel(store_id=self.store_id, pred_week=self.pred_week),
            PrepareTestData(store_id=self.store_id, pred_week=self.pred_week),
        ]

    def run(self):
        model = pickle.load(
            open(
                config.output_dir
                / f"model_store_{self.store_id}_week_{self.pred_week}.pickle",
                "rb",
            )
        )
        X_test = pd.read_pickle(
            config.output_dir
            / f"test_store_{self.store_id}_week_{self.pred_week}.pickle"
        )
        print(f"cols test: {len(X_test.columns)}")

        # TODO: assert X_test = sales.drop(config.pred_target_col,
        #       pred_date_col not in columns

        X_test = X_test.drop("date", axis=1)

        y_test = model.predict(X_test)
        preds = pd.DataFrame(y_test)
        preds.to_pickle(
            config.output_dir
            / f"pred_store_{self.store_id}_week_{self.pred_week}.pickle"
        )

        # Clean-up files that are not needed anymore
        os.remove(
            config.output_dir
            / f"train_store_{self.store_id}_week_{self.pred_week}.pickle"
        )

    def output(self):
        return luigi.LocalTarget(
            str(
                config.output_dir
                / f"pred_store_{self.store_id}_week_{self.pred_week}.pickle"
            )
        )


class RunPredictionStore(luigi.WrapperTask):
    store_id = luigi.Parameter()
    pred_week_list = config.pred_week_list

    def requires(self):
        return {
            w: RunPredictionStoreWeek(store_id=self.store_id, pred_week=w)
            for w in self.pred_week_list
        }

    def run(self):
        os.remove(
            config.output_dir / f"features_sales_base_store_{self.store_id}.pickle"
        )
        os.remove(config.output_dir / f"features_sales_ts_store_{self.store_id}.pickle")


class RunPredictionAll(luigi.Task):
    store_list = config.store_list

    def requires(self):
        return {s: RunPredictionStore(store_id=s) for s in self.store_list}

    def run(self):
        preds = []
        for store_id in self.store_list:
            for pred_week in config.pred_week_list:
                preds.append(
                    pd.read_pickle(
                        config.output_dir
                        / f"pred_store_{store_id}_week_{pred_week}.pickle"
                    )
                )

        preds_all = pd.concat(preds)
        preds_all.to_pickle(config.output_dir / "pred.pickle")

    def output(self):
        return luigi.LocalTarget(str(config.output_dir / "pred.pickle"))


class RunPipeline(luigi.Task):
    start_time = datetime.now()

    def requires(self):
        return RunPredictionAll()

    def run(self):
        print("Running pipeline is done: {}".format(datetime.now() - self.start_time))
