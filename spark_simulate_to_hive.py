
####################################################################################################
#
#   Spark Simulate Dataset and Write to Hive as ORC (with Snappy Compression)
#
#   This script will simulate data, execute basic commands (counts(), show(), ect.)
#
#   Usage: 
#   /usr/hdp/current/spark2-client/bin/spark-submit --master yarn --deploy-mode client --driver-memory 20G --conf "spark.dynamicAllocation.enabled=true" --conf "spark.shuffle.service.enabled=true" --conf "spark.dynamicAllocation.initialExecutors=6" --conf "spark.dynamicAllocation.minExecutors=6" --conf "spark.yarn.executor.memoryOverhead=8G" --conf "spark.yarn.driver.memoryOverhead=8G" spark_simulate_to_hive.py <number_of_records, default=50 million> <table_name>
#   This will also run from the pyspark shell
#
####################################################################################################

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import datetime
import requests
import json
import sys

try:
    number_of_records   = int(sys.argv[1])
    tablename           = sys.argv[2]
except:
    number_of_records   = 5000
    tablename           = 'sim_table1'

spark = SparkSession \
    .builder \
    .enableHiveSupport() \
    .getOrCreate()

start_time = datetime.datetime.now()
start_time_total = start_time

df = spark.range(0,number_of_records)

def sim_random():
    import random
    return random.random()

def sim_rate():
    import random
    return random.random() * random.triangular(100,1000,100)

def sim_bool1():
    import random
    return random.choice(['TRUE']*2 + ['FALSE']*8)

def sim_bool2():
    import random
    return int(random.choice([1]*2 + [0]*8))

def sim_gender():
    import random
    return random.choice(['MALE']*3 + ['FEMALE']*6)

def sim_age():
    import random
    return int(random.triangular(15,90,35))

def sim_rating():
    import random
    return int(random.triangular(1,10,6))

def sim_state():
    import random
    return ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'][int(random.triangular(0,49,2))]

def sim_name():
    import random
    return ['Frank','Dean','Dan','Sammy','James','Andrew','Scott','Steven','Kristen','Stephen','William','Craig','Stacy','Stuart','Christopher','Alan','Megan','Brian','Kevin','Kate','Molly','Derek','Martin','Thomas','Neil','Barry','Ian','Ashley','Iain','Gordon','Alexander','Graeme','Peter','Darren','Graham','George','Kenneth','Allan','Lauren','Douglas','Keith','Lee','Katy','Grant','Ross','Jonathan','Gavin','Nicholas','Joseph','Stewart'][int(random.triangular(0,49,2))]

def sim_date():
    import random
    return random.choice(['2015','2016','2017']) + '-' + str(random.choice(range(1,13))).zfill(2) + '-' + str(random.choice(range(1,30))).zfill(2)

udf_random  = udf(sim_random, FloatType())
udf_rate    = udf(sim_rate, FloatType())
udf_bool1   = udf(sim_bool1, StringType())
udf_bool2   = udf(sim_bool2, IntegerType())
udf_gender  = udf(sim_gender, StringType())
udf_age     = udf(sim_age, IntegerType())
udf_state   = udf(sim_state, StringType())
udf_name    = udf(sim_name, StringType())
udf_date    = udf(sim_date, StringType())
udf_rating  = udf(sim_rating, IntegerType())

sim = df.withColumn('date', udf_date() ) \
  .withColumn('name', udf_name() ) \
  .withColumn('age', udf_age() ) \
  .withColumn('gender', udf_gender() ) \
  .withColumn('state', udf_state() ) \
  .withColumn('flag1', udf_bool1() ) \
  .withColumn('flag2', udf_bool2() ) \
  .withColumn('flag3', udf_bool2() ) \
  .withColumn('metric1', udf_random() ) \
  .withColumn('metric2', udf_random() ) \
  .withColumn('metric3', udf_random() ) \
  .withColumn('fee', udf_random() ) \
  .withColumn('rate', udf_random() ) \
  .withColumn('rating', udf_rating() )

print '\n\n' + '#'*100
print '#'
print '#    Spark Performance Check - Results'
print '#'
print '#'*100

runtime_msg = '\n[ INFO ] Simulated ' + str(number_of_records) + ' records (' + str(len(sim.columns)) + ' columns) in ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds\n'
print runtime_msg

sim.createOrReplaceTempView(tablename)
spark.sql('CREATE TABLE ' + str(tablename) + ' STORED AS ORC tblproperties ("orc.compress" = "SNAPPY") AS SELECT * from ' + str(tablename))


#ZEND
