# Databricks notebook source
# DBTITLE 1,ML pipeline: read, split, train, log
import os
import warnings
import logging

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from mlflow.models import infer_signature

# CONFIG
TABLE = "some_catalog.new_york_taxi.yellow_trip_data_silver"
target_col = "fare_amount"
feature_cols = [
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_min"
]

UC_TMP = "/Volumes/some_catalog/new_york_taxi/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = UC_TMP

# SUPPRESS WARNINGS / LOGGING
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

spark = SparkSession.getActiveSession()

# LOAD
df = spark.read.table(TABLE)

for c in feature_cols:
    df = df.withColumn(c, F.col(c).cast("double"))

df = df.dropna(subset=feature_cols + [target_col])

# split
train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)

# pipeline
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
lr = LinearRegression(featuresCol="features", labelCol=target_col)
pipeline = Pipeline(stages=[assembler, lr])

if mlflow.active_run():
    mlflow.end_run()

conda_env = {
    "name": "mlflow-spark-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12",
        "pip",
        {"pip": ["pyspark==4.0.0", "mlflow"]}
    ],
}

with mlflow.start_run():
    fitted = pipeline.fit(train)
    test_pred = fitted.transform(test)

    rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse").evaluate(test_pred)
    mae  = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae").evaluate(test_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_param("target_col", target_col)
    mlflow.log_param("features", feature_cols)

    # Signature (input and output)
    sample_in  = train.select(feature_cols).limit(200).toPandas()
    sample_out = fitted.transform(train.limit(200)).select("prediction").toPandas()
    signature = infer_signature(sample_in, sample_out)

    mlflow.spark.log_model(
        fitted,
        artifact_path="model",
        dfs_tmpdir=UC_TMP,
        signature=signature,
        conda_env=conda_env
    )

    print("MLflow run complete. Metrics and model logged.")


# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning: Hyperopt + MLflow
import mlflow
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from mlflow.models import infer_signature

TABLE = "some_catalog.new_york_taxi.yellow_trip_data_silver"
target_col = "fare_amount"
feature_cols = [
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_min"
]
UC_TMP = "/Volumes/some_catalog/new_york_taxi/mlflow_tmp"

spark = SparkSession.getActiveSession()
df = spark.read.table(TABLE)
for c in feature_cols:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.dropna(subset=feature_cols + [target_col])
train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)

# Hyperopt search space
search_space = {
    'regParam': hp.uniform('regParam', 0.0, 0.2),
    'elasticNetParam': hp.uniform('elasticNetParam', 0.0, 1.0)
}

# Objective function
def objective(params):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    lr = LinearRegression(featuresCol="features", labelCol=target_col,
                         regParam=params['regParam'],
                         elasticNetParam=params['elasticNetParam'])
    pipeline = Pipeline(stages=[assembler, lr])
    with mlflow.start_run(nested=True):
        fitted = pipeline.fit(train)
        val_pred = fitted.transform(val)
        rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse").evaluate(val_pred)
        mlflow.log_params(params)
        mlflow.log_metric("val_rmse", rmse)
        # Add signature for UC registration
        sample_in = train.select(feature_cols).limit(200).toPandas()
        sample_out = fitted.transform(train.limit(200)).select("prediction").toPandas()
        signature = infer_signature(sample_in, sample_out)
        mlflow.spark.log_model(fitted, "model", dfs_tmpdir=UC_TMP, signature=signature)
    return {'loss': rmse, 'status': STATUS_OK}

# Run Hyperopt
trials = Trials()
best = fmin(fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)

print(f"Best hyperparameters: {best}")


# COMMAND ----------

# DBTITLE 1,Register best model in MLflow Model Registry
import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "some_catalog.new_york_taxi.fare_amount_regression"
RUN_ID = "4f72b78654d34a51b66d65c7ad691969"
MODEL_URI = f"runs:/{RUN_ID}/model"

UC_TMP = "/Volumes/some_catalog/new_york_taxi/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = UC_TMP

client = MlflowClient()

# Create registered model
client.get_registered_model(MODEL_NAME)

# Create model version
mv = client.create_model_version(
    name=MODEL_NAME,
    source=MODEL_URI,
    run_id=RUN_ID
)

# Set alias Champion
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=mv.version
)

print(f"✅ Model registered")
print(f"Model name: {MODEL_NAME}")
print(f"Version: {mv.version}")
print("Alias: Champion")


# COMMAND ----------

import os
import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

UC_TMP = "/Volumes/some_catalog/new_york_taxi/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = UC_TMP

# Minimal fix: define train, feature_cols, final_model
TABLE = "some_catalog.new_york_taxi.yellow_trip_data_silver"
feature_cols = [
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_min"
]
target_col = "fare_amount"

spark = SparkSession.getActiveSession()
df = spark.read.table(TABLE)
for c in feature_cols:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.dropna(subset=feature_cols + [target_col])
train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
lr = LinearRegression(featuresCol="features", labelCol=target_col)
pipeline = Pipeline(stages=[assembler, lr])
final_model = pipeline.fit(train)

# signature (обов’язково)
sample_in  = train.select(feature_cols).limit(200).toPandas()
sample_out = final_model.transform(train.limit(200)).select("prediction").toPandas()
signature = infer_signature(sample_in, sample_out)

pip_reqs = [
    "mlflow==2.22.0",
    "pyspark==4.0.0"
]

with mlflow.start_run(run_name="fare_amount_spark_pipreqs"):
    mlflow.spark.log_model(
        final_model,
        artifact_path="model",
        dfs_tmpdir=UC_TMP,
        signature=signature,
        pip_requirements=pip_reqs
    )
    run_id = mlflow.active_run().info.run_id
    print("NEW RUN:", run_id)


# COMMAND ----------

import os
from mlflow.tracking import MlflowClient

MODEL_NAME = "some_catalog.new_york_taxi.fare_amount_regression"
RUN_ID = "b99f28dfaac04354be3eb0d0d1455a14"
MODEL_URI = f"runs:/{RUN_ID}/model"

UC_TMP = "/Volumes/some_catalog/new_york_taxi/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = UC_TMP

client = MlflowClient()

# create model if missing
try:
    client.get_registered_model(MODEL_NAME)
except Exception:
    client.create_registered_model(MODEL_NAME)

# create new version
mv = client.create_model_version(
    name=MODEL_NAME,
    source=MODEL_URI,
    run_id=RUN_ID
)

# move Champion alias to this version
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=mv.version
)

print("Registered version:", mv.version)
print("Champion -> version:", mv.version)
None


# COMMAND ----------

import os
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

UC_TMP = "/Volumes/some_catalog/new_york_taxi/mlflow_tmp"
os.environ["MLFLOW_DFS_TMP"] = UC_TMP

TABLE = "some_catalog.new_york_taxi.yellow_trip_data_silver"
feature_cols = ["VendorID","passenger_count","trip_distance","RatecodeID","PULocationID","DOLocationID","trip_duration_min"]
target_col = "fare_amount"

spark = SparkSession.getActiveSession()
df = spark.read.table(TABLE)
for c in feature_cols:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.dropna(subset=feature_cols + [target_col])

train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)
trainval = train.unionByName(val)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
lr = LinearRegression(featuresCol="features", labelCol=target_col)
pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(trainval)

lr_stage = model.stages[-1]
coef = np.array(lr_stage.coefficients)
intercept = float(lr_stage.intercept)

class LRPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, coef, intercept, feature_cols):
        self.coef = coef
        self.intercept = intercept
        self.feature_cols = feature_cols

    def predict(self, context, model_input: pd.DataFrame):
        X = model_input[self.feature_cols].astype(float).to_numpy()
        y = X @ self.coef + self.intercept
        return pd.DataFrame({"prediction": y})

py_model = LRPyfunc(coef=coef, intercept=intercept, feature_cols=feature_cols)

sample_in = trainval.select(feature_cols).limit(200).toPandas()
sample_out = py_model.predict(None, sample_in)
signature = infer_signature(sample_in, sample_out)

with mlflow.start_run(run_name="fare_amount_pyfunc_serving"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=py_model,
        signature=signature,
        pip_requirements=["mlflow==2.22.0", "pandas", "numpy"]
    )
    new_run = mlflow.active_run().info.run_id
    print("NEW RUN (pyfunc):", new_run)


# COMMAND ----------

# DBTITLE 1,Compare serving prediction with real value
import requests, json
from pyspark.sql import SparkSession

# Вибрати реальний рядок із датасету
spark = SparkSession.getActiveSession()
df = spark.read.table("some_catalog.new_york_taxi.yellow_trip_data_silver")
row = df.select(
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_min", "fare_amount"
).dropna().limit(1).collect()[0]

features = [
    int(row[0]), int(row[1]), float(row[2]), int(row[3]), int(row[4]), int(row[5]), float(row[6])
]

real_fare = float(row[7])

payload = {
    "dataframe_split": {
        "columns": ["VendorID","passenger_count","trip_distance","RatecodeID","PULocationID","DOLocationID","trip_duration_min"],
        "data": [features]
    }
}

URL = "https://dbc-35a3b90e-ef96.cloud.databricks.com/serving-endpoints/fare_amount_regression/invocations"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
r = requests.post(URL, headers=headers, data=json.dumps(payload))

print("Features:", features)
print("Real fare_amount:", real_fare)
print("Predicted:", r.text)


# COMMAND ----------

# DBTITLE 1,Batch scoring: Registry → Delta
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import timedelta

# CONFIG
MODEL_NAME   = "some_catalog.new_york_taxi.fare_amount_regression"
MODEL_ALIAS  = "Champion"
PRED_TABLE   = "some_catalog.new_york_taxi.fare_amount_predictions"
SOURCE_TABLE = "some_catalog.new_york_taxi.yellow_trip_data_silver"

feature_cols = [
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_min"
]

spark = SparkSession.getActiveSession()

# 1) Load model from Registry (alias Champion)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# 2) Pick a REAL scoring window from the dataset itself (last 1 day relative to max_dt)
bounds = spark.read.table(SOURCE_TABLE).select(
    F.max("tpep_pickup_datetime").alias("max_dt")
).collect()[0]

max_dt = bounds["max_dt"]
start_dt = max_dt - timedelta(days=1)

df_new = spark.read.table(SOURCE_TABLE).where(
    (F.col("tpep_pickup_datetime") >= F.lit(start_dt)) &
    (F.col("tpep_pickup_datetime") <  F.lit(max_dt))
).select(feature_cols + ["tpep_pickup_datetime"])

n = df_new.limit(1).count()
print("Scoring range:", start_dt, "->", max_dt)

if n == 0:
    print("No data to score in this range.")
else:
    # 3) Batch scoring (Spark -> pandas -> pyfunc predict)
    pdf = df_new.toPandas()
    pdf["prediction"] = model.predict(pdf)

    # 4) Write predictions to Delta table
    pred_df = spark.createDataFrame(pdf)
    pred_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(PRED_TABLE)

    print(f"Batch scoring complete. Rows scored: {len(pdf)}. Written to {PRED_TABLE}")

# 5) Quick check
spark.read.table(PRED_TABLE).orderBy("tpep_pickup_datetime", ascending=False).limit(20).display()
None


# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.getActiveSession()

TABLE = "some_catalog.new_york_taxi.yellow_trip_data_silver"
target_col = "fare_amount"
feature_cols = [
    "VendorID", "passenger_count", "trip_distance", "RatecodeID",
    "PULocationID", "DOLocationID", "trip_duration_min"
]

# ---------- Load & basic cleaning ----------
df = spark.read.table(TABLE)
for c in feature_cols + [target_col]:
    df = df.withColumn(c, F.col(c).cast("double"))

df = df.dropna(subset=feature_cols + [target_col])

# ---------- Train/Val/Test split ----------
train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)

# ---------- Train model ----------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
lr = LinearRegression(featuresCol="features", labelCol=target_col)
pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(train)

# ---------- Evaluate ----------
pred_test = model.transform(test)

rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse").evaluate(pred_test)
mae  = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae").evaluate(pred_test)

print("TEST RMSE:", rmse)
print("TEST MAE :", mae)

# ---------- Sample for plots ----------
sample = (pred_test
          .select(target_col, "prediction", "trip_distance", "trip_duration_min")
          .orderBy(F.rand(seed=42))
          .limit(20000)
          .toPandas())

# 1) Fare distribution
plt.figure()
plt.hist(sample[target_col].clip(lower=0, upper=200), bins=60)
plt.title("Distribution of fare_amount (clipped to [0, 200])")
plt.xlabel("fare_amount")
plt.ylabel("count")
plt.show()

# 2) Actual vs Predicted
plt.figure()
plt.scatter(sample[target_col], sample["prediction"], s=5)
plt.title("Actual vs Predicted (sample)")
plt.xlabel("actual fare_amount")
plt.ylabel("predicted fare_amount")
plt.show()

# 3) Residuals distribution
resid = sample["prediction"] - sample[target_col]
plt.figure()
plt.hist(resid.clip(lower=-100, upper=100), bins=80)
plt.title("Residuals (prediction - actual), clipped to [-100, 100]")
plt.xlabel("residual")
plt.ylabel("count")
plt.show()

# 4) (Optional) Coefficients bar chart (interpretable slide)
lr_stage = model.stages[-1]  # LinearRegressionModel
coefs = np.array(lr_stage.coefficients)
coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs}).sort_values("coef")

plt.figure()
plt.barh(coef_df["feature"], coef_df["coef"])
plt.title("Linear Regression Coefficients")
plt.xlabel("coefficient value")
plt.ylabel("feature")
plt.show()
