

import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id, col, expr, when, concat, lit, udf, split
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor, LinearRegression, GeneralizedLinearRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#######################################################################################
#
#   Load Data
#
#######################################################################################

rawdata = spark.read.load('/nyc_taxi_data.csv', format="csv", header=True, inferSchema=True)

rawdata.show(10)
print '\nTotal Records: ' + str(rawdata.count()) + '\n'
for i in rawdata.dtypes: print i


#######################################################################################
#
#   Explore Data
#
#######################################################################################

# How many unique cars are there (based on vehicle_id)
rawdata.select('vehicle_id').distinct().count()


# Describe / find basic stats for numerical data
rawdata.describe(['trip_distance','passenger_count','payment_amount']).show()


# Option 1 - What are my top earning cars
rawdata.groupBy('vehicle_id') \
       .agg({'payment_amount': 'sum'}) \
       .sort("sum(payment_amount)", ascending=False) \
       .show()


# Option 2 - What are my top earning cars
rawdata.createOrReplaceTempView("rawdata_sql")
spark.sql("SELECT vehicle_id, sum(payment_amount) as sum FROM rawdata_sql group by vehicle_id order by sum desc").show()


# What is the average distance and average payment by car/vehicle
spark.sql("SELECT vehicle_id, mean(trip_distance) as avg_distance, mean(payment_amount) as avg_payment FROM rawdata_sql group by vehicle_id order by avg_distance desc").show()


# Covariance
rawdata.stat.cov('payment_amount', 'fare_amount')
rawdata.stat.cov('payment_amount', 'fare_amount')

# Correlation
rawdata.stat.corr('payment_amount', 'fare_amount')
rawdata.stat.corr('tip_amount', 'fare_amount')

# CrossTab
#rawdata.stat.crosstab("item1", "item2").show()


# Frequent Items
#freq = rawdata.stat.freqItems(["a", "b", "c"], 0.4)  # Find freqent items up to 40%
#freq.collect()[0]


#######################################################################################
#
#   Transformations
#
#######################################################################################

# Extract Trip time
def time_delta(pickup_time, dropoff_time):
    pickup_time_out  = datetime.datetime.strptime(pickup_time, '%m/%d/%y %H:%M')
    dropoff_time_out = datetime.datetime.strptime(dropoff_time, '%m/%d/%y %H:%M')
    trip_time        = (dropoff_time_out - pickup_time_out).seconds / float(60)
    return trip_time

f = udf(time_delta, FloatType())

# (1) Calculate "trip_time"
# (2) Create a "tip_flag" for any record where a customer leaves a tip
# (3) Extract the Pickup Day (as an integer)
# (4) Extract the Pickup Hour (as an integer)
transformed1 = rawdata.withColumn("trip_time", f(rawdata.pickup_datetime, rawdata.dropoff_datetime)) \
                      .withColumn("tip_flag", (when(rawdata.tip_amount > 0.0, 1).otherwise(0)) ) \
                      .withColumn("pickup_day", split(rawdata.pickup_datetime,"\/")[1].cast("integer") ) \
                      .withColumn("pickup_hour", split(split(rawdata.pickup_datetime," ")[1],":")[0].cast("integer") )


#######################################################################################
#
#   Model Prep
#
#######################################################################################

# String Indexer
strindexer = StringIndexer(inputCol="vehicle_id", outputCol="vehicle_id_index")
modelprep1 = strindexer.fit(transformed1).transform(transformed1)

features = ['pickup_longitude','passenger_count','tolls_amount','tip_amount','trip_distance']

# Generate Features Vector and Label
va = VectorAssembler(inputCols=features, outputCol="features")
modelprep2 = va.transform(modelprep1).withColumn("label", col("trip_time"))



#######################################################################################
#
#   Define TRAINING and TESTING dataframes
#
#######################################################################################

# Random Spliting
training, testing = modelprep2.randomSplit([0.8, 0.2])

#modelprep2.count()
#training.count()
#testing.count()


#######################################################################################
#
#   Modeling - GLM (Regression)
#
#######################################################################################

glm = GeneralizedLinearRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.3)
glmmodel = glm.fit(training)

summary = glmmodel.summary

# Show Coefficients and Intercept
print("\nFeatures: " + str(features) + "\n")
print("\nCoefficients: " + str(glmmodel.coefficients) + "\n")
print("\nIntercept: " + str(glmmodel.intercept) + "\n")
print("\nPValues: " + str(summary.pValues) + "\n")

# Summarize the model over the training set and print out some metrics
#print("\nCoefficient Standard Errors: " + str(summary.coefficientStandardErrors))
#print("T Values: " + str(summary.tValues))
#print("P Values: " + str(summary.pValues))
#print("Dispersion: " + str(summary.dispersion))
#print("Null Deviance: " + str(summary.nullDeviance))
#print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
#print("Deviance: " + str(summary.deviance))
#print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
#print("AIC: " + str(summary.aic))
#print("Deviance Residuals: ")
#summary.residuals().show()

# Make predictions.
predictions = glmmodel.transform(testing)

# Select example rows to display.
predictions.select("prediction", "label").show(30,False)

evaluator = RegressionEvaluator(metricName="rmse")  # rmse (default)|mse|r2|mae
RMSE = evaluator.evaluate(predictions)
print 'RMSE: ' + str(RMSE)



#######################################################################################
#
#   Modeling - Gradient Boosting (Regression)
#
#######################################################################################

gbt = GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)
#gbt = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)

gbtmodel = gbt.fit(training)

# Make predictions.
predictions = gbtmodel.transform(testing)

# Select example rows to display.
predictions.select("prediction", "label").show(30,False)

evaluator = RegressionEvaluator(metricName="rmse")  # rmse (default)|mse|r2|mae
RMSE = evaluator.evaluate(predictions)
print 'RMSE: ' + str(RMSE)


#######################################################################################
#
#   Modeling - Gradient Boosting (Classifier)
#
#######################################################################################

va = VectorAssembler(inputCols=['passenger_count','tolls_amount','fare_amount'], outputCol="features")
modelprep2 = va.transform(modelprep1).withColumn("label", col("tip_flag"))

training, testing = modelprep2.randomSplit([0.8, 0.2])

gbt = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)

gbtmodel = gbt.fit(training)

# Make predictions.
predictions = gbtmodel.transform(testing)

# Select example rows to display.
predictions.select("prediction", "label").show(30,False)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")   # f1|weightedPrecision|weightedRecall|accuracy
accuracy = evaluator.evaluate(predictions)
print("Accuracy = " + str(accuracy))





#ZEND