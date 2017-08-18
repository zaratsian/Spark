
#######################################################################################################################
#
#   Spark Analysis on the MovieLens Dataset (20 Million Movie Reviews)
#
#   Download Movie Dataset from here: https://grouplens.org/datasets/movielens/
#   Here's the direct link: http://files.grouplens.org/datasets/movielens/ml-latest.zip
#
#   Additional info and background can be found here as well: https://www.kaggle.com/grouplens/movielens-20m-dataset
#
#   Datasets contains 26 million reviews (Stored in Hive as ORC using https://github.com/zaratsian/Apache-Hive/blob/master/movielens.sql)
#       1.)  movies that contains movie information
#       2.)  ratings that contains ratings of movies by users
#       3.)  links that contains identifiers that can be used to link to other sources
#       4.)  tags that contains tags applied to movies by users
#
#   Developed on: 
#       Spark 2.1.1.2.6.1.0-129 
#       Python 2.7.5
#
#######################################################################################################################

import datetime
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, split, explode
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, GeneralizedLinearRegression, LinearRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession \
    .builder \
    .appName("pyspark_movielens") \
    .enableHiveSupport() \
    .getOrCreate()

start_time = datetime.datetime.now()

# Load Data from Hive
#movies  = spark.sql("SELECT * FROM movies")
#ratings = spark.sql("SELECT * FROM ratings")
#links   = spark.sql("SELECT * FROM links")
#tags    = spark.sql("SELECT * FROM tags")

# Load Data from HDFS (as CSV)
movies  = spark.read.load("/ml-latest/movies.csv", "csv", delimiter=",", inferSchema=True, header=True)
ratings = spark.read.load("/ml-latest/ratings.csv", "csv", delimiter=",", inferSchema=True, header=True)
links   = spark.read.load("/ml-latest/links.csv", "csv", delimiter=",", inferSchema=True, header=True)
tags    = spark.read.load("/ml-latest/tags.csv", "csv", delimiter=",", inferSchema=True, header=True)

print '[ INFO ] Number of Records in movies:  ' + str(movies.count())
print '[ INFO ] Number of Records in ratings: ' + str(ratings.count())
print '[ INFO ] Number of Records in links:   ' + str(links.count())
print '[ INFO ] Number of Records in tags:    ' + str(tags.count())

# Pyspark Dataframe Joins
join1 = ratings.join(movies, ratings.movieId == movies.movieId).drop(movies.movieId)
print '[ INFO ] Number of Records in join1:   ' + str(join1.count())
join1.show()

join2 = join1.join(tags, (join1.movieId == tags.movieId) & (join1.userId == tags.userId), how='leftouter').drop(movies.movieId).drop(tags.movieId).drop(tags.userId).drop(tags.timestamp)
print '[ INFO ] Number of Records in join2:   ' + str(join2.count())
join2.show()

genres = join2.select(explode(split(col('genres'), '\|'))).distinct()
genres.show()
genres = genres.collect()
genres = [genre[0].encode('utf-8') for genre in genres]

def extract_genres(genres_string):
    return [1 if genre in genres_string.split('|') else 0 for genre in genres]

udf_extract_genres = udf(extract_genres, ArrayType(IntegerType()))

#join2.select(['userid','movieid', udf_extract_genres(col('genres'))]).show()

enriched1 = join2.withColumn('genres_vector', udf_extract_genres(col('genres'))) \
                 .withColumn(genres[0].lower(), col('genres_vector')[0]) \
                 .withColumn(genres[1].lower(), col('genres_vector')[1]) \
                 .withColumn(genres[2].lower(), col('genres_vector')[2]) \
                 .withColumn(genres[3].lower(), col('genres_vector')[3]) \
                 .withColumn(genres[4].lower(), col('genres_vector')[4]) \
                 .withColumn(genres[5].lower(), col('genres_vector')[5]) \
                 .withColumn(genres[6].lower(), col('genres_vector')[6]) \
                 .withColumn(genres[7].lower(), col('genres_vector')[7]) \
                 .withColumn(genres[8].lower(), col('genres_vector')[8]) \
                 .withColumn(genres[9].lower(), col('genres_vector')[9]) \
                 .withColumn(genres[10].lower(), col('genres_vector')[10]) \
                 .withColumn(genres[11].lower(), col('genres_vector')[11]) \
                 .withColumn(genres[12].lower(), col('genres_vector')[12]) \
                 .withColumn(genres[13].lower(), col('genres_vector')[13]) \
                 .withColumn(genres[14].lower(), col('genres_vector')[14]) \
                 .withColumn(genres[15].lower(), col('genres_vector')[15]) \
                 .withColumn(genres[16].lower(), col('genres_vector')[16]) \
                 .withColumn(genres[17].lower(), col('genres_vector')[17]) \
                 .withColumn(genres[18].lower(), col('genres_vector')[18]) \
                 .withColumn(genres[19].lower(), col('genres_vector')[19]) \
                 .drop('genres', 'genres_vector')

print '[ INFO ] Listing columns...'
print ','.join(enriched1.columns)
print '\r\n'
print '[ INFO ] Printing Data Types...'
for datatype in enriched1.dtypes:
    print datatype

var_target   = 'rating'
var_features = [col for col in enriched1.columns if col not in ['userId','movieId','rating','timestamp','title','tag']]

# Generate Features Vector and Label
va = VectorAssembler(inputCols=var_features, outputCol="features")

modelprep1 = va.transform(enriched1).select('userId','movieId','rating','features')

training, testing, other = modelprep1.randomSplit([0.07, 0.03, 0.90])

print '[ INFO ] Training:          ' + str(training.count()) + ' records'
print '[ INFO ] Testing:           ' + str(training.count()) + ' records'

gb = GBTRegressor(featuresCol="features", labelCol=var_target, predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)

gbmodel = gb.fit(training)
#gbmodel.save('/tmp/spark_models/kaggle_bike_sharing_gb_model')

predictions = gbmodel.transform(testing)

print '[ INFO ] Printing predictions vs label...'
predictions.show(10,False).select('prediction',var_target)

evaluator = RegressionEvaluator(labelCol=var_target, predictionCol="prediction")
print '[ INFO ] Model Fit (RMSE):  ' + str(evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
#print '[ INFO ] Model Fit (MSE):   ' + str(evaluator.evaluate(predictions, {evaluator.metricName: "mse"}))
#print '[ INFO ] Model Fit (R2):    ' + str(evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))

total_runtime_seconds = (datetime.datetime.now() - start_time).seconds

print '#'*100
print '[ INFO ] Total Runtime:     ' + str(total_runtime_seconds) + ' seconds'
print '#'*100


#ZEND
