# Databricks notebook source
# MAGIC %pip install folium

# COMMAND ----------

import folium

athenes_map = folium.Map([37.97580099999999, 23.737850000000016], zoom_start=14)
url = "http://data.insideairbnb.com/greece/attica/athens/2023-09-21/visualisations/neighbourhoods.geojson"
folium.GeoJson(url).add_to(athenes_map)

# COMMAND ----------

athenes_map

# COMMAND ----------

listings_df = spark.read.table('airbnb.raw.listings')
display(listings_df)

# COMMAND ----------

listings_locations = listings_df.select('price', 'latitude', 'longitude').toPandas().to_dict(orient='records')
listings_locations[:10]

# COMMAND ----------

from folium.plugins import MarkerCluster
from folium import Marker

marker_cluster = MarkerCluster()
for row in listings_locations[:30]: # remove [:30] to see all listings on map
    marker = Marker(location=[row['latitude'], row['longitude']], popup=row['price'], icon = folium.Icon(color='blue',icon='ok-sign'))
    marker_cluster.add_child(marker)

athenes_map.add_child(marker_cluster)
