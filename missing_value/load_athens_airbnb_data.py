# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Load data
# MAGIC
# MAGIC **source**:  Airbnb data for Athens at http://insideairbnb.com/get-the-data
# MAGIC
# MAGIC **destination**: airbnb.raw schema 

# COMMAND ----------

# MAGIC %run ./setup

# COMMAND ----------

import io
import pandas as pd
import requests

def load_data_from_url_as_spark_df(url, **params):
    """Loads data from url. Optional keyword arguments are passed to pandas.read_csv."""
    response = requests.get(url)
    dx = pd.read_csv(io.BytesIO(response.content), **params)  
    return spark.createDataFrame(dx)

# COMMAND ----------

import json 
import pyspark.sql.functions as f
from pyspark.sql.types import StringType, TimestampType
from delta.tables import DeltaTable

def update_schema_with_metadata_fields(schema):
    """Helper method to add metadata fields to contact schema."""
    return schema\
            .add("processing_datetime", TimestampType(), True)\
            .add("area", StringType(), True)
        
def add_metadata_columns(df, area: str):
    """Helper method to add metadata columns to dataframe."""
    return (
        df.withColumn("processing_datetime", f.current_timestamp())
        .withColumn("area", f.lit(area))
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load Listings

# COMMAND ----------

# load contract schema 
with open(contracts_path + "/listing_schema.json", "r") as fin:
    listing_schema = StructType.fromJson(json.loads(fin.read()))

# update contract schema with metadata fields
raw_listing_schema = update_schema_with_metadata_fields(listing_schema)

# initialize delta table
listing_delta = (
    DeltaTable.createIfNotExists(spark)
    .addColumns(raw_listing_schema)
    .tableName('airbnb.raw.listings')
).execute()

# COMMAND ----------

listings_url = "http://data.insideairbnb.com/greece/attica/athens/2023-09-21/data/listings.csv.gz"

listings_df = load_data_from_url_as_spark_df(
    listings_url, sep=',', index_col=0, quotechar='"', compression='gzip'
)

# COMMAND ----------

_ = (listings_df
     .transform(lambda x: add_metadata_columns(x, "Athens"))
     .writeTo("airbnb.raw.listings")
     .append())

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load calendar data 

# COMMAND ----------

# load contract schema 
with open(contracts_path + "/calendar_schema.json", "r") as fin:
    calendar_schema = StructType.fromJson(json.loads(fin.read()))

# update contract schema with metadata fields
raw_calendar_schema = update_schema_with_metadata_fields(calendar_schema)

# initialize delta table
calendar_delta = (
    DeltaTable.createIfNotExists(spark)
    .addColumns(raw_calendar_schema)
    .tableName('airbnb.raw.calendar')
).execute()

# COMMAND ----------

callendar_url = "http://data.insideairbnb.com/greece/attica/athens/2023-09-21/data/calendar.csv.gz"

callendar_df = load_data_from_url_as_spark_df(
    callendar_url, sep=',', index_col=0, quotechar='"', compression='gzip'
)

# COMMAND ----------

_ = (callendar_df
     .transform(lambda x: add_metadata_columns(x, "Athens"))
     .writeTo("airbnb.raw.calendar")
     .append())

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load agg_listings

# COMMAND ----------

# load contract schema 
with open(contracts_path + "/agg_listing_schema.json", "r") as fin:
    agg_listing_schema = StructType.fromJson(json.loads(fin.read()))

# update contract schema with metadata fields
raw_agg_listing_schema = update_schema_with_metadata_fields(agg_listing_schema)

# initialize delta table
calendar_delta = (
    DeltaTable.createIfNotExists(spark)
    .addColumns(raw_agg_listing_schema)
    .tableName('airbnb.raw.agg_listing')
).execute()

# COMMAND ----------

agg_listing_url = "http://data.insideairbnb.com/greece/attica/athens/2023-09-21/visualisations/listings.csv"

agg_listing_df = load_data_from_url_as_spark_df(
    agg_listing_url, sep=',', index_col=0, quotechar='"'
)


# COMMAND ----------

_ = (agg_listing_df
     .transform(lambda x: add_metadata_columns(x, "Athens"))
     .writeTo("airbnb.raw.agg_listing")
     .append())

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Load geo json data 
# MAGIC
# MAGIC In orde to load geo data we can use pandas-geojson package. 
# MAGIC But w need to install it first. Plase ```%pip install pandas-geojson ```in the first cell in notebook and rerul entire notebook. 
# MAGIC
# MAGIC ```
# MAGIC   import json
# MAGIC   from pandas_geojson import read_geojson_url
# MAGIC
# MAGIC   neighbourhoods_geo_url = "http://data.insideairbnb.com/greece/attica/athens/2023-09-21/visualisations/neighbourhoods.geojson"
# MAGIC
# MAGIC   # load geo data & save as json file.
# MAGIC   geo_json = read_geojson_url(neighbourhoods_geo_url)
# MAGIC   dbutils.fs.put(neighbourhoods_geo_path, json.dumps(geo_json))
# MAGIC
# MAGIC   # load neighbourhoods_geo as spark data frame 
# MAGIC   neighbourhoods_geo_df = spark.read.format('json').load(neighbourhoods_geo_path)
# MAGIC ```
