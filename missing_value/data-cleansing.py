# Databricks notebook source
# DBTITLE 0,--i18n-8c6d3ef3-e44b-4292-a0d3-1aaba0198525
# MAGIC %md 
# MAGIC
# MAGIC # Data Exploration 
# MAGIC
# MAGIC Imagine that we would like to build a ML model to predict the price of a listing based on lodaded data in raw area. 
# MAGIC
# MAGIC Our primary goal at this stage is to conduct data exploration to identify and resolve potential issues with the data, such as invalid types, missing values, and anomalies. 
# MAGIC
# MAGIC Please run <code>load_athens_airbnb_data</code> notebook to get dataset into <code>airbnb.raw</code> schema

# COMMAND ----------

# DBTITLE 0,--i18n-969507ea-bffc-4255-9a99-2306a594625f
# MAGIC %md 
# MAGIC
# MAGIC ## Load Listings Dataset
# MAGIC
# MAGIC Let's load the Airbnb Athens listing dataset in.

# COMMAND ----------


raw_df = spark.read.table('airbnb.raw.listings')
display(raw_df)

# COMMAND ----------

raw_df.columns

# COMMAND ----------

# DBTITLE 0,--i18n-94856418-c319-4915-a73e-5728fcd44101
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC For the sake of simplicity, only keep certain columns from this dataset.

# COMMAND ----------

columns_to_keep = [
    'host_id','host_since','host_is_superhost','latitude','longitude','property_type','room_type','accommodates','bathrooms','bathrooms_text','bedrooms','beds','amenities','price','minimum_nights','maximum_nights', 'number_of_reviews','review_scores_rating','license','instant_bookable','reviews_per_month'
]

base_df = raw_df.select(columns_to_keep)
display(base_df)

# COMMAND ----------

# DBTITLE 0,--i18n-a12c5a59-ad1c-4542-8695-d822ec10c4ca
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC  
# MAGIC ## Fixing Data Types
# MAGIC
# MAGIC Take a look at the schema above. You'll notice that the **`price`** field got picked up as string. For our task, we need it to be a numeric (double type) field. 
# MAGIC
# MAGIC Let's fix that.

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-4ad08138-4563-4a93-b038-801832c9bc73
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Summary statistics
# MAGIC
# MAGIC Two options:
# MAGIC * **`describe`**: count, mean, stddev, min, max
# MAGIC * **`summary`**: describe + interquartile range (IQR)
# MAGIC
# MAGIC **Question:** When to use IQR/median over mean? Vice versa?

# COMMAND ----------

display(fixed_price_df.describe())

# COMMAND ----------

display(fixed_price_df.summary())

# COMMAND ----------

# DBTITLE 0,--i18n-bd55efda-86d0-4584-a6fc-ef4f221b2872
# MAGIC %md 
# MAGIC
# MAGIC ### Explore Dataset with Data Profile
# MAGIC
# MAGIC The **Data Profile** feature in Databricks notebooks offers valuable insights and benefits for data analysis and exploration. By leveraging Data Profile, users gain a comprehensive overview of their **dataset's characteristics, statistics, and data quality metrics**. This feature enables data scientists and analysts to understand the data distribution, identify missing values, detect outliers, and explore descriptive statistics efficiently.
# MAGIC
# MAGIC There are two ways of viewing Data Profiler. The first option is the UI.
# MAGIC
# MAGIC - After using `display` function to show a data frame, click **+** icon next to the *Table* in the header. 
# MAGIC - Click **Data Profile**. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC This functionality is also available through the dbutils API in Python, Scala, and R, using the dbutils.data.summarize(df) command. We can also use **`dbutils.data.summarize(df)`** to display Data Profile UI.
# MAGIC
# MAGIC Note that this features will profile the entire data set in the data frame or SQL query results, not just the portion displayed in the table

# COMMAND ----------

dbutils.data.summarize(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-e9860f92-2fbe-4d23-b728-678a7bb4734e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Extreme values
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the **`price`** column.

# COMMAND ----------

display(fixed_price_df.select("price").describe())

# COMMAND ----------

display(fixed_price_df.select('price'))

# COMMAND ----------

# DBTITLE 0,--i18n-4a8fe21b-1dac-4edf-a0a3-204f170b05c9
# MAGIC %md 
# MAGIC There are some super-expensive listings, and it's up to theSubject Matter Experts to decide what to do with them.
# MAGIC Let's see first how many listings we can find where the *price* is extream.

# COMMAND ----------

fixed_price_df.filter(col("price") >= 10000).count()

# COMMAND ----------

# DBTITLE 0,--i18n-bf195d9b-ea4d-4a3e-8b61-372be8eec327
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC We have 1 listing that has extream values and lets remove it.

# COMMAND ----------

pos_prices_df = fixed_price_df.filter(col("price") < 10000)

# COMMAND ----------

# DBTITLE 0,--i18n-dc8600db-ebd1-4110-bfb1-ce555bc95245
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

# COMMAND ----------

display(pos_prices_df)

# COMMAND ----------

display(pos_prices_df.select("minimum_nights").describe())

# COMMAND ----------

display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

# DBTITLE 0,--i18n-5aa4dfa8-d9a1-42e2-9060-a5dcc3513a0d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC A minimum stay of 90 days seems to be a reasonable limit here due to . Let's filter out those records where the *minimum_nights* is greater than 90.

# COMMAND ----------

min_nights_df = pos_prices_df.filter(col("minimum_nights") <= 90)

display(min_nights_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Fix bathrooms
# MAGIC
# MAGIC It seems like bathroom column contains null values and information regarding bathrooms counts contained in bathrooms_text column.

# COMMAND ----------

display(min_nights_df.select('bathrooms', 'bathrooms_text'))

# COMMAND ----------


from pyspark.sql.functions import element_at, split

# this code will fail because of anusual data points
# please uncoment and try to figure out hoe to fix this. 

bathrooms_df = min_nights_df


# bathrooms_df = (
#   bathrooms_df
#   .withColumn('bathrooms', element_at(split('bathrooms_text', ' '), 1).cast('double'))
#   .drop('bathrooms_text')
# )

# display(bathrooms_df)



# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Fix boolean columns types 
# MAGIC
# MAGIC There are two columns `host_is_superhost` and `instant_bookable` that potentially may be usefull, but they are encoded as 't' or 'f' values.
# MAGIC Lets change it to boolean type. 
# MAGIC

# COMMAND ----------

boolean_df = (
  bathrooms_df
  .withColumn('instant_bookable', col('instant_bookable') == 't')
  .withColumn('host_is_superhost', col('host_is_superhost') == 't')
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Fix amenities
# MAGIC
# MAGIC We can notice that `amenities` column contains a list of available items but it looks like string list which is not easy to work with. 
# MAGIC PySpark has a number of build-in methods to work with arrays: __array_*__
# MAGIC Lets convert amenities into propper aray of strings column.

# COMMAND ----------


from pyspark.sql.functions import explode, lower

amenities_df = boolean_df.withColumn('amenities', split(translate('amenities', '\\]\\[\\"', ''), ','))

display(
  amenities_df
  .select(explode('amenities'))
  .withColumn('item', lower('col'))
  .groupBy('item').count()
  .sort('count')
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC ## Combine all above transformations
# MAGIC

# COMMAND ----------

def prepare_athens_listings(raw_df):
  from pyspark.sql.functions import col, translate, element_at, split, to_date
  columns_to_keep = [
      'host_id','host_since','host_is_superhost','latitude','longitude','property_type','room_type','accommodates','bathrooms','bathrooms_text','bedrooms','beds','amenities','price','minimum_nights','maximum_nights', 'number_of_reviews','review_scores_rating','license','instant_bookable','reviews_per_month'
      # any other columns you may find useful for our future model? 
  ]
  df = raw_df.select(*columns_to_keep)


  # to do
  # add validation to identify any unusual data points and mark them as invalid. please add reason why those datapoints are invalid. 
  
  return df 

df = prepare_athens_listings(raw_df)
display(df)

# COMMAND ----------

# DBTITLE 0,--i18n-25a35390-d716-43ad-8f51-7e7690e1c913
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Handling Null Values
# MAGIC
# MAGIC There are a lot of different ways to handle null values. Sometimes, null can actually be a key indicator of the thing you are trying to predict (e.g. if you don't fill in certain portions of a form, probability of it getting approved decreases).
# MAGIC
# MAGIC Some ways to handle nulls:
# MAGIC * Drop any records that contain nulls
# MAGIC * Numeric:
# MAGIC   * Replace them with mean/median/zero/etc.
# MAGIC * Categorical:
# MAGIC   * Replace them with the mode
# MAGIC   * Create a special category for null
# MAGIC   
# MAGIC **If you do ANY imputation techniques for categorical/numerical features, you MUST include an additional field specifying that field was imputed.**

# COMMAND ----------

# to do: 
# Work in team and brainstorm the best way to visualize missing values in dataset 

# COMMAND ----------

# to do: 

# create a new schema called 'validatated' in airbnb catalog and save your validated listing dataset.
