

################################################################################################################################
#
#   PySpark Structured Streaming with Kafka 0.10
#
#   Batch Example
#
################################################################################################################################
'''
Usage:

/usr/hdp/current/spark2-client/bin/pyspark \
    --master yarn \
    --deploy-mode client \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.2.0

/usr/hdp/current/spark2-client/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.2.0 \
    pyspark_structured_stream_kafka_batch.py
'''

import os,sys
import datetime,time
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, window, udf, desc, asc
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .appName("StructuredStream_with_Kafka") \
    .getOrCreate()

events = spark \
        .read \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "dzaratsian2.field.hortonworks.com:6667") \
        .option("subscribe", "clickstream") \
        .load()

events = events.selectExpr("CAST(value AS STRING)")

# Option #1: Parse by column, using withColumn
parsed_events = events.withColumn('uid', split(events['value'],',')[0].cast(StringType()) )         \
                    .withColumn('user', split(events['value'],',')[1].cast(StringType()) )          \
                    .withColumn('datetime', split(events['value'],',')[2].cast(TimestampType()) )   \
                    .withColumn('state', split(events['value'],',')[3].cast(StringType()) )         \
                    .withColumn('duration', split(events['value'],',')[4].cast(FloatType()) )       \
                    .withColumn('rate', split(events['value'],',')[5].cast(FloatType()) )           \
                    .withColumn('action', split(events['value'],',')[6].cast(StringType()) )

parsed_events.show(10,False)



print('\n\n[ INFO ] Displaying user count. 60 second window with 15 sec sliding duration...\n\n')

# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.window
# pyspark.sql.functions.window(timeColumn, windowDuration, slideDuration=None, startTime=None)
windowedCounts = parsed_events.groupBy(
    window(parsed_events.datetime, "1 minutes", "15 seconds"),
    parsed_events.user) \
    .count()

windowedCounts.sort(desc("count")).show(10,False)



print('\n\n[ INFO ] Displaying average duration by user. 60 second window with 15 sec sliding duration...\n\n')

windowedAvg = parsed_events.groupBy(
    window(parsed_events.datetime, "1 minutes", "15 seconds"),
    parsed_events.user) \
    .agg({'duration': 'mean'})

windowedAvg.sort(desc("avg(duration)")).show(10,False)


'''

/usr/hdp/current/kafka-broker/bin/kafka-topics.sh --create --zookeeper dzaratsian2.field.hortonworks.com:2181 --replication-factor 1 --partitions 1 --topic clickstream 

/usr/hdp/current/kafka-broker/bin/kafka-topics.sh --delete --zookeeper dzaratsian2.field.hortonworks.com:2181 --topic clickstream 

/usr/hdp/current/kafka-broker/bin/kafka-topics.sh --zookeeper dzaratsian2.field.hortonworks.com:2181 --list 

/usr/hdp/current/kafka-broker/bin/kafka-console-consumer.sh --zookeeper dzaratsian2.field.hortonworks.com:2181 --topic clickstream --from-beginning

'''

#ZEND
