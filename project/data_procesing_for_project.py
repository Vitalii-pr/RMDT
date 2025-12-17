# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE some_catalog.new_york_taxi.yellow_trips_raw
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   current_timestamp() AS _ingest_ts
# MAGIC FROM some_catalog.new_york_taxi.yellow_trip_data_feb_2025;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS bronze_rows
# MAGIC FROM some_catalog.new_york_taxi.yellow_trips_raw;
# MAGIC
# MAGIC SELECT *
# MAGIC FROM some_catalog.new_york_taxi.yellow_trips_raw
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW v_yellow_dq AS
# MAGIC SELECT
# MAGIC   CAST(VendorID AS INT) AS vendor_id,
# MAGIC   tpep_pickup_datetime AS pickup_ts,
# MAGIC   tpep_dropoff_datetime AS dropoff_ts,
# MAGIC   CAST(passenger_count AS INT) AS passenger_count,
# MAGIC   trip_distance,
# MAGIC   CAST(PULocationID AS INT) AS pu_location_id,
# MAGIC   CAST(DOLocationID AS INT) AS do_location_id,
# MAGIC   CAST(payment_type AS INT) AS payment_type,
# MAGIC   total_amount,
# MAGIC   tip_amount,
# MAGIC
# MAGIC   to_date(tpep_pickup_datetime) AS pickup_date,
# MAGIC   hour(tpep_pickup_datetime) AS pickup_hour,
# MAGIC
# MAGIC   (unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime)) / 60.0 AS trip_duration_min,
# MAGIC
# MAGIC   sha2(
# MAGIC     concat_ws('||',
# MAGIC       CAST(VendorID AS STRING),
# MAGIC       CAST(tpep_pickup_datetime AS STRING),
# MAGIC       CAST(PULocationID AS STRING),
# MAGIC       CAST(DOLocationID AS STRING)
# MAGIC     ),
# MAGIC     256
# MAGIC   ) AS trip_id,
# MAGIC
# MAGIC   _ingest_ts,
# MAGIC
# MAGIC   array_remove(array(
# MAGIC     CASE WHEN tpep_pickup_datetime IS NULL THEN 'NULL_PICKUP_TS' END,
# MAGIC     CASE WHEN tpep_dropoff_datetime IS NULL THEN 'NULL_DROPOFF_TS' END,
# MAGIC     CASE WHEN tpep_dropoff_datetime < tpep_pickup_datetime THEN 'NEG_DURATION' END,
# MAGIC     CASE WHEN trip_distance < 0 THEN 'NEG_DISTANCE' END,
# MAGIC     CASE WHEN total_amount < 0 THEN 'NEG_TOTAL' END
# MAGIC   ), NULL) AS reject_reasons
# MAGIC
# MAGIC FROM some_catalog.new_york_taxi.yellow_trips_raw;
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

bronze_df = spark.table(
    "some_catalog.new_york_taxi.yellow_trip_data_feb_2025"
)

silver_df = (
    bronze_df
    .withColumn(
        "trip_duration_min",
        (F.col("tpep_dropoff_datetime").cast("long") -
         F.col("tpep_pickup_datetime").cast("long")) / 60
    )
    .filter(F.col("trip_distance") >= 0)
    .filter(F.col("total_amount") >= 0)
    .filter(F.col("trip_duration_min") > 0)
)

silver_df.write.mode("overwrite").saveAsTable(
    "some_catalog.new_york_taxi.yellow_trip_data_silver"
)

print("SILVER ROW COUNT:", silver_df.count())
display(silver_df.limit(5))
display(silver_df.orderBy(F.desc("tpep_pickup_datetime")).limit(5))


# COMMAND ----------

silver_df = spark.table(
    "some_catalog.new_york_taxi.yellow_trip_data_silver"
)

gold_df = (
    silver_df
    .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
    .groupBy("pickup_date")
    .agg(
        F.count("*").alias("trips_cnt"),
        F.round(F.avg("trip_distance"), 2).alias("avg_trip_distance"),
        F.round(F.avg("total_amount"), 2).alias("avg_total_amount"),
        F.round(F.sum("total_amount"), 2).alias("total_revenue")
    )
)

gold_df.write.mode("overwrite").saveAsTable(
    "some_catalog.new_york_taxi.yellow_trip_data_gold"
)

print("GOLD ROW COUNT:", gold_df.count())
display(gold_df.limit(5))
display(gold_df.orderBy(F.desc("pickup_date")).limit(5))

# COMMAND ----------

spark.sql("""
SELECT 
  (SELECT COUNT(*) FROM some_catalog.new_york_taxi.yellow_trips_raw) AS bronze_cnt,
  (SELECT COUNT(*) FROM some_catalog.new_york_taxi.yellow_trip_data_silver) AS silver_cnt
""").show()

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.dq_numeric_stats;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.dq_reject_reasons;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.dq_run_metrics;
# MAGIC
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_payment_type_mix;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_peak_hours;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_revenue_daily;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_tips_by_hour_zone;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_top_pickup_zones;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_trip_length_segments;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.gold_trips_by_zone;
# MAGIC
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.yellow_trip_data_feb_2025;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.yellow_trips_quarantine;
# MAGIC DROP TABLE IF EXISTS some_catalog.new_york_taxi.yellow_trips_silver;
# MAGIC