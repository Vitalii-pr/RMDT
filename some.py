import matplotlib as plt
from pyspark.sql import functions as F

raw_df = spark.read.table("airbnb.raw.listings")
display(raw_df)

total_rows = raw_df.count()


missing_exprs = [
    (F.count(F.when(F.col(c).isNull(), c)) / total_rows).alias(c)
    for c in raw_df.columns
]

missing_df = raw_df.select(missing_exprs)
display(missing_df)
data = []
for c in raw_df.columns:
    null_count = raw_df.filter(F.col(c).isNull()).count()
    data.append((c, null_count, null_count / total_rows))

missing_stats = spark.createDataFrame(data, ["column", "missing_count", "missing_rate"])
display(missing_stats)

import matplotlib.pyplot as plt

pdf = missing_stats.toPandas().sort_values("missing_rate", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(pdf["column"], pdf["missing_rate"], color="#007acc")
plt.gca().invert_yaxis()
plt.xlabel("Missing rate")
plt.title("Missing Values per Column in Airbnb Dataset")
plt.show()

