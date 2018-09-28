
########################################################################################################
#
#   PySpark Basic Model Flow (used to show model deployment strategies)
#
#   Note: The goal of this code is not to produce the best possible model, but rather to 
#         show the framework for a pyspark model pipeline, which we will then deploy into production.
#
########################################################################################################

'''
Usage:
/usr/hdp/current/spark2-client/bin/pyspark --master yarn --deploy-mode client --driver-memory 4G --conf "spark.dynamicAllocation.enabled=true" --conf "spark.shuffle.service.enabled=true" --conf "spark.dynamicAllocation.initialExecutors=6"
'''

import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, LinearRegression, GeneralizedLinearRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator

rawdata = spark.read.load('hdfs:///tmp/loan_200k_new.csv', format="csv", header=True, inferSchema=True)

rawdata.show(10,False)
print '\nTotal Records: ' + str(rawdata.count()) + '\n'
for i in rawdata.dtypes: print i

rawdata.groupby(rawdata.purpose).count().show(20,False)
rawdata.groupby(rawdata.default).count().show(20,False)

#training, testing = rawdata.randomSplit([0.80, 0.20])

df_defaults  = rawdata.filter(rawdata.default==1)
df_nodefault = rawdata.filter(rawdata.default==0)

training_defaults, testing_defaults = df_defaults.randomSplit([0.80, 0.20])
default_pct = training_defaults.count() / float(df_nodefault.count())
training_nodefault, testing_nodefault = df_nodefault.randomSplit([default_pct, (1.0 - default_pct)])

training = training_defaults.unionAll(training_nodefault)
testing  = testing_defaults.unionAll(testing_nodefault)

si  = StringIndexer(inputCol="purpose", outputCol="purpose_index")
hot = OneHotEncoder(inputCol="purpose_index", outputCol="purpose_features")
va  = VectorAssembler(inputCols=["loan_amnt", "interest_rate", "employment_length", "home_owner", "income", "verified", "open_accts", "credit_debt", "purpose_features"], outputCol="features")
dtr = DecisionTreeRegressor(featuresCol="features", labelCol="default", predictionCol="prediction", maxDepth=2, varianceCol="variance")
gbr = GBTRegressor(featuresCol="features", labelCol="default", predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)
gbc = GBTClassifier(featuresCol="features", labelCol="default", predictionCol="prediction", maxDepth=5, maxIter=20, seed=12345)

pipeline = Pipeline(stages=[si, hot, va, gbc])

model = pipeline.fit(training)
model.write().overwrite().save('hdfs:///tmp/spark_model')

predictions = model.transform(testing)

predictions.select(['default','prediction']).sort(col('prediction').desc()).show(25,False)

#evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="default")
#rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
#r2   = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

#evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="default")
#evaluator.evaluate(predictions)
#evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="default")
evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})


#ZEND
