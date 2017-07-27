
############################################################################################
#
#   PySpark (Basic) Sentiment Analysis
#
############################################################################################

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
import re
from textblob import TextBlob


spark = SparkSession \
    .builder \
    .appName("PySpark Text Analytics") \
    .enableHiveSupport() \
    .getOrCreate()

# Read Data from Hive
#rawdata = spark.sql("SELECT * FROM hotel_reviews")

rawdata = spark.read.csv("/tomorrowland.csv", inferSchema=True, header=True)

rawdata.show(50)

'''
rawdata_list =[
    (100, 1,  'this is the best product, i love it.'),
    (101, 0,  'this has been a great experience but the product was not satisfactory.'),
    (102, 1,  'the product is awesome.'),
    (103, -1, 'i hate the product and the price is terrible'),
    (104, 1,  'the product works great and it is awesome to use'),
    (105, -1, 'customer service was helpful but the product is expensive and it is bad')
    ]

rawdata = spark.createDataFrame(rawdata_list, ['id','lable','text'])
'''

def zsentiment(text):
    return TextBlob(text).sentiment.polarity


udf_cleantext = udf(zsentiment , FloatType())
text_variable = 'text'
clean_text    = rawdata.withColumn("sentiment_score", udf_cleantext( rawdata[text_variable] ))
clean_text.show(10,False)


#ZEND
