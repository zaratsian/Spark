
####################################################################################################
#
#   Spark Performance Check
#
#   This script will simulate data, execute basic commands (counts(), show(), ect.) and collect 
#   cluster setting and runtime stats that can be used for Spark tuninig and configuration.
#
#   Usage: ./bin/pyspark spark_performance_check.py
#
#   Output results will be written to /tmp/spark_performance_check.txt
#
####################################################################################################

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import datetime
import requests
import json

conf  = SparkConf().setAppName('Spark Performance Check').setMaster('yarn').set('deploy-mode','client')
sc    = SparkContext(conf=conf)
spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .getOrCreate()

file = '/tmp/spark_performance_check.txt'
output_file = open(file,'wb')

start_time = datetime.datetime.now()
start_time_total = start_time

number_of_records = 50000000

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
  .withColumn('metric1', udf_random() ) \
  .withColumn('metric1', udf_random() ) \
  .withColumn('metric1', udf_random() ) \
  .withColumn('rating', udf_rating() )

print '\n\n' + '#'*100
print '#'
print '#    Spark Performance Check - Results'
print '#'
print '#'*100 + '\n\n'

runtime_msg = '[ INFO ] Simulated ' + str(number_of_records) + ' records (' + str(len(sim.columns)) + ' columns) in ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds\n'
print runtime_msg
output_file.write(runtime_msg)

#start_time = datetime.datetime.now()
#sim.show()
#runtime_msg = '[ INFO ] show() Runtime: ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds\n'
#print runtime_msg
#output_file.write(runtime_msg)

start_time = datetime.datetime.now()
sim.count()
runtime_msg = '[ INFO ] count() Runtime: ' + str((datetime.datetime.now() - start_time).seconds) + ' seconds\n'
print runtime_msg
output_file.write(runtime_msg)

output_file.write('\n\nSPARK METRICS:\n')
for metric in sc._conf.getAll():
    output_file.write(str(metric))
    output_file.write('\n')

req = requests.get('http://localhost:4040/api/v1/applications')

appid = json.loads(req.content)[0]['id']
output_file.write('[ INFO ] Spark App ID: ' + str(appid))
output_file.write('\n')

#req = requests.get('http://localhost:4040/api/v1/applications/' + str(appid) + '/environment')

output_file.write('[ INFO ] Spark Executors:')
req = requests.get('http://localhost:4040/api/v1/applications/' + str(appid) + '/executors')
runtime_msg = str(req.content)
output_file.write(runtime_msg)
output_file.write('\n')

output_file.write('[ INFO ] Spark Jobs:')
req = requests.get('http://localhost:4040/api/v1/applications/' + str(appid) + '/jobs')
runtime_msg = str(req.content)
output_file.write(runtime_msg)
output_file.write('\n')

jobid = json.loads(req.content)[0]['jobId']

import py4j.protocol  
from py4j.protocol import Py4JJavaError  
from py4j.java_gateway import JavaObject  
from py4j.java_collections import JavaArray, JavaList

from pyspark import RDD, SparkContext  
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer

# Helper function to convert python object to Java objects
def _to_java_object_rdd(rdd):  
    """
    Return a JavaRDD of Object by unpickling
    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

JavaObj = _to_java_object_rdd(sim.rdd)

# Now we can run the estimator
output_file.write('[ INFO ] Estimated size (in bytes): ' + str(sc._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)))

output_file.write('\n[ INFO ] Total Runtime: ' + str((datetime.datetime.now() - start_time_total).seconds) + ' seconds\n')

output_file.close()

print '\n\n' + '#'*100
print '#'
print '#    Complete - Results can be found at ' + str(file)
print '#'
print '#'*100 + '\n\n'

#ZEND
