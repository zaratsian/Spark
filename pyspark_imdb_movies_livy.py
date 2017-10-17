
##################################################################################################
'''
NOTES:

1.) Install Anaconda across nodes

2.) Set PYSPARK_PYTHON env var:
sudo su
su livy
cd ~
echo "" >> ~/.bashrc
echo "export PYSPARK_PYTHON=/opt/anaconda2/bin/python2.7" >> ~/.bashrc

3.) Set spark.yarn.appMasterEnv.PYSPARK_PYTHON = /opt/anaconda2/bin/python2.7

'''
##################################################################################################
#
#   Livy Server - Spark Example (REST API)
#
##################################################################################################


import requests
import json
import datetime,time


##################################################################################################
#
#   Function:  Initial Session and Submit Code
#
##################################################################################################

'''
host='dzaratsian5.field.hortonworks.com'
port='8999'
code_payload = {'code': "spark.range(0,10).count()"}
session_id=''
'''


def spark_livy_interactive(host='', port='8999', code_payload='', session_id=''):
    
    result      = ''
    base_url    = 'http://' + str(host) +':'+ str(port)
    session_url = base_url + '/sessions'
    headers     = {'Content-Type': 'application/json', 'X-Requested-By': 'spark'}
    
    if session_id == '':
        #data = {'kind': 'pyspark'}
        data = {'kind': 'pyspark', 'conf':{'spark.yarn.appMasterEnv.PYSPARK_PYTHON':'/opt/anaconda2/bin/python2.7'}}
        spark_session = requests.post(base_url + '/sessions', data=json.dumps(data), headers=headers)
        if spark_session.status_code == 201:
            session_id  = spark_session.json()['id']
            session_url = base_url + spark_session.headers['location']
            print '[ INFO ] Status Code:           ' + str(spark_session.status_code)
            print '[ INFO ] Session State:         ' + str(spark_session.json()['state'])
            print '[ INFO ] Session ID:            ' + str(session_id)
            print '[ INFO ] Payload:               ' + str(spark_session.content)
            
            # Loop until Spark Session is ready (i.e. In the Idle State)
            session_state = ''
            while (session_state == '') or (session_state == 'starting'):
                time.sleep(0.25)
                print '[ INFO ] Session State:         ' + str(session_state)
                session = requests.get(session_url, headers=headers)
                if session.status_code == 200:
                    session_state = session.json()['state']
                else:
                    print '[ ERROR ] Status Code: ' + str(session.status_code)
                    session_state = 'end'
            
            print '[ INFO ] Session State:         ' + str(session_state)
            print '[ INFO ] Spark App ID:          ' + str(session.json()['appId'])
            print '[ INFO ] Spark App URL:         ' + str(session.json()['appInfo']['sparkUiUrl'])
        else:
            print '[ ERROR ] Failed to start Spark Session, Status Code: ' + str(spark_session.status_code)
    else:
        print '[ INFO ] Using existing Session ID: ' + str(session_id)
    
    try:
        submit_code = requests.post(base_url + '/sessions/' + str(session_id) + '/statements', data=json.dumps(code_payload), headers=headers)
    except:
        result = {'state':'Error with code submission'}
        return session_id, result
    
    if submit_code.status_code == 201:
        submit_code_state = ''
        while (submit_code_state == '') or (submit_code_state == 'running'):
            time.sleep(0.25)
            code_response = requests.get(base_url + submit_code.headers['location'], headers=headers)
            if code_response.status_code == 200:
                result = code_response.json()
                submit_code_state = code_response.json()['state']
                print '[ INFO ] Code Submit State:     ' + str(submit_code_state)
                #print '\n' + '#'*50
                #print '[ INFO ] Result:   ' + str(result)
                #print '#'*50
            else:
                print '[ ERROR ] Status Code:   ' + str(code_response.status_code)
    else:
        print '[ ERROR ] Failed to submit Spark code successfully, Status Code: ' + str(submit_code.status_code)
     
    return session_id, result


##################################################################################################
#
#   Function:  Delete Session
#
##################################################################################################

def delete_spark_livy_session(host='', port='', session_id=''):
    print '[ INFO ] Deleting Session ' + str(session_id) + '...'
    time.sleep(1)
    base_url    = 'http://' + str(host) +':'+ str(port)
    headers     = {'Content-Type': 'application/json', 'X-Requested-By': 'spark'}
    del_request = requests.delete(base_url + '/sessions/' + str(session_id), headers=headers)
    if del_request.status_code == 200: print '[ INFO ] Successfully deleted Session ' + str(session_id)


##################################################################################################
#
#   Test Function
#
##################################################################################################

code_payload = {'code': '''

import os

os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda2/bin/python2.7"
os.environ["PYSPARK_PYTHON"]        = "/opt/anaconda2/bin/python2.7"

import datetime, time 
import re, random, sys
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, StringType, FloatType, LongType
from pyspark.sql.functions import struct, array, lit, monotonically_increasing_id, col, expr, when, concat, udf, split, size
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor, LinearRegression, GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.ml.feature import Word2Vec
#import nltk
#import spacy 
#nlp = spacy.load("en")
#import sparknlp.annotators as sparknlp
#sc.setLogLevel("ERROR")

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("pyspark_livy_movie_app_dz") \
    .getOrCreate()

############################################################################################################
#
#   Load Rawdata
#   Source: https://www.kaggle.com/tmdb/tmdb-movie-metadata
#
############################################################################################################

schema = StructType([                                           \
    StructField("budget", IntegerType(), True),                 \
    StructField("genres", StringType(), True),                  \
    StructField("homepage", StringType(), True),                \
    StructField("id", StringType(), True),                      \
    StructField("keywords", StringType(), True),                \
    StructField("original_language", StringType(), True),       \
    StructField("original_title", StringType(), True),          \
    StructField("overview", StringType(), True),                \
    StructField("popularity", FloatType(), True),               \
    StructField("production_companies", StringType(), True),    \
    StructField("production_countries", StringType(), True),    \
    StructField("release_date", StringType(), True),            \
    StructField("revenue", LongType(), True),                   \
    StructField("runtime", IntegerType(), True),                \
    StructField("spoken_languages", StringType(), True),        \
    StructField("status", StringType(), True),                  \
    StructField("tagline", StringType(), True),                 \
    StructField("title", StringType(), True),                   \
    StructField("vote_average", FloatType(), True),             \
    StructField("vote_count", IntegerType(), True)])

rawdata = spark.read.format('csv').load('hdfs://dzaratsian0.field.hortonworks.com:8020/tmp/tmdb_5000_movies.csv', format="csv", header=True, schema=schema, quote='"', escape='"', mode="DROPMALFORMED")

############################################################################################################
#
#   Extract Genres from JSON
#
############################################################################################################

def extract_genre(column_name):
    import json
    try:
        genre = json.loads(column_name)[0]['name'].strip()
        
        if (genre=='Action') or (genre=='Adventure'):
            genre = 'Action/Adventure'
        elif (genre=='Fantasy') or (genre=='Science Fiction'):
            genre = 'Science Fiction'
        elif (genre=='Horror') or (genre=='Thriller'):
            genre = 'Horror'
        elif (genre=='Drama'):
            genre = 'Drama'
        elif (genre=='Comedy'):
            genre = 'Comedy'
        else:
            genre = 'Other'
        
        return genre
    except:
        return 'Other'

udf_extract_genre = udf(extract_genre, StringType())

enriched1 = rawdata.withColumn("genre", udf_extract_genre('genres'))

#enriched1.cache
#enriched1.count()
#enriched1.select(['genres','genre']).show(20,False)
#enriched1.groupBy('genre').count().show(50,False)

############################################################################################################
#
#   Transformations
#
############################################################################################################

json_columns = ['genres','keywords','production_companies','production_countries','spoken_languages']
text_columns = ['original_title','overview','tagline','title']
drop_columns = json_columns + ['original_title','tagline','popularity','release_date','status','original_language','homepage']

# Filter DF, drop where revenue == 0
enriched2 = enriched1.filter('revenue != 0')

# Extract Date/Time Variables
def extract_year(date_var):
    try:
        return datetime.datetime.strptime(date_var, '%Y-%m-%d').year
    except:
        return random.randint(2005,2016)

def extract_month(date_var):
    try:
        return datetime.datetime.strptime(date_var, '%Y-%m-%d').month
    except:
        return random.randint(1,12)

def extract_day(date_var):
    try:
        return datetime.datetime.strptime(date_var, '%Y-%m-%d').day
    except:
        return random.randint(1,28)

udf_year  = udf(extract_year, IntegerType())
udf_month = udf(extract_month, IntegerType())
udf_day   = udf(extract_day, IntegerType())

enriched2 = enriched2.withColumn("year", udf_year('release_date')) \
       .withColumn("month", udf_month('release_date')) \
       .withColumn("day", udf_day('release_date'))

def bucket_revenue(revenue):
    if revenue <= 10000000:
        return 'Less than $10M'
    elif 10000000 < revenue <= 25000000:
        return '$10M - $25M'
    elif 25000000 < revenue <= 50000000:
        return '$25M - $50M'
    elif 50000000 < revenue <= 100000000:
        return '$50M - $100M'
    elif 100000000 < revenue <= 200000000:
        return '$100M - $200M'
    elif revenue > 200000000:
        return 'Over $200M'
    else:
        return -1

udf_bucket_revenue  = udf(bucket_revenue, StringType())

enriched2 = enriched2.withColumn("revenue_bucket", udf_bucket_revenue('revenue'))
enriched2 = enriched2.withColumn('revenue_over_100M', when(col("revenue") >= 100000000, 1).otherwise(0))

def bucket_vote(vote_average):
    if vote_average >= 7.0:
        return 'Great'
    elif 6.0 <= vote_average < 7.0:
        return 'Good'
    elif 5.0 <= vote_average < 6.0:
        return 'Ok'
    elif vote_average < 5.0:
        return 'Bad'
    else:
        return 'No Vote'

udf_bucket_vote  = udf(bucket_vote, StringType())

enriched2 = enriched2.withColumn("vote_bucket", udf_bucket_vote('vote_average'))

# Drop Columns
for c in drop_columns:
    enriched2 = enriched2.drop(c)

#enriched2.cache()
enriched2.count()

############################################################################################################
#
#   Model Pipeline (similar to above) - Regression
#
############################################################################################################

target   = 'revenue'
features = ['budget','runtime','vote_average','vote_count','year','month','day','genre_index']

# Model Input Variables: ( budget|runtime|vote_average|vote_count|year|month|day )

training, testing = enriched2.randomSplit([0.8, 0.2], seed=12345)

si  = StringIndexer(inputCol="genre", outputCol="genre_index")
va  = VectorAssembler(inputCols=features, outputCol="features")
gbr = GBTRegressor(featuresCol="features", labelCol=target, predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)

pipeline = Pipeline(stages=[si, va, gbr])

mlmodel     = pipeline.fit(training)
predictions = mlmodel.transform(testing)

evaluator = RegressionEvaluator(metricName="rmse", labelCol=target)  # rmse (default)|mse|r2|mae
RMSE = evaluator.evaluate(predictions)
#print 'RMSE: ' + str(RMSE)

evaluator = RegressionEvaluator(metricName="r2", labelCol=target)
R2 = evaluator.evaluate(predictions)
#print 'R2:   ' + str(R2)

mlmodel.save('hdfs://dzaratsian0.field.hortonworks.com:8020/tmp/model_predict_movie_revenue')

'''}


#code_payload = {'code': ''' spark.range(0,10).count() '''}


session_id, result = spark_livy_interactive(host='dzaratsian4.field.hortonworks.com', port='8999', code_payload=code_payload, session_id='')
session_id
result
result['output']['data']['text/plain']


code_payload = {'code': '''

spark.range(0,50).count()

'''}

session_id, result = spark_livy_interactive(host='dzaratsian4.field.hortonworks.com', port='8999', code_payload=code_payload, session_id=session_id)
session_id
result
result['output']['data']['text/plain']


delete_spark_livy_session(host='dzaratsian4.field.hortonworks.com', port='8999', session_id='10')


#ZEND
