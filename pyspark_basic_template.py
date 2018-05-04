

from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id, col, expr, when, concat, lit, udf, split


spark = SparkSession \
    .builder \
    .appName("spark_validation") \
    .config("hive.exec.dynamic.partition", "true") \
    .config("hive.exec.dynamic.partition.mode", "nonstrict") \
    .enableHiveSupport() \
    .getOrCreate()


# Load data from HDFS
rawdata = spark.read.load('/tmp/nyc_taxi_data.csv', format="csv", header=True, inferSchema=True)


# Display results
rawdata.show(10,False)


# Parse Fixed Length file
rawdata.select(
    rawdata.value.substr(1,3).alias('id'),
    rawdata.value.substr(4,8).alias('date'),
    rawdata.value.substr(12,3).alias('string'),
    rawdata.value.substr(15,4).cast('integer').alias('integer')
).show(10,False)


# How many unique cars are there (based on vehicle_id)
rawdata.select('vehicle_id').distinct().count()


# Describe / find basic stats for numerical data
rawdata.describe(['trip_distance','passenger_count','payment_amount']).show()


# Option 1 - What are my top earning cars
rawdata.groupBy('vehicle_id') \
       .agg({'payment_amount': 'sum'}) \
       .sort("sum(payment_amount)", ascending=False) \
       .show()

df.agg({"age": "max"}).collect()


# Option 2 - What are my top earning cars
rawdata.createOrReplaceTempView("rawdata_sql")
spark.sql("SELECT vehicle_id, sum(payment_amount) as sum FROM rawdata_sql group by vehicle_id order by sum desc").show()


# Write spark DF to Hive (stored as ORC)
enriched_data \
        .write.format("orc") \
        .partitionBy("var1","var2") \
        .mode("overwrite") \
        .saveAsTable("myhivetable")


#ZEND
