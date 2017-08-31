
############################################################################################################
#
#   PySpark Image Classification
#   
'''
Download Data:
wget https://github.com/zsellami/images_classification/blob/master/personalities.zip

Usage:
/spark/bin/pyspark --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11

pip install nose
pip install pillow
pip install keras
pip install h5py
pip install py4j
pip install tensorflow
'''
#
#   https://github.com/databricks/spark-deep-learning
#   https://medium.com/linagora-engineering/making-image-classification-simple-with-spark-deep-learning-f654a8b876b8
#
############################################################################################################

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from sparkdl import readImages

img_dir = "/tmp/personalities/"

jobs_df = readImages(img_dir + "/jobs").withColumn("label", lit(1))
zuck_df = readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))

training_pct = 0.70
testing_pct  = 0.30

jobs_train, jobs_test = jobs_df.randomSplit([training_pct, testing_pct])
zuck_train, zuck_test = zuck_df.randomSplit([training_pct, testing_pct])

train_df = jobs_train.unionAll(zuck_train)
print '[ INFO ] Number of Training Records: ' + str(train_df.count())

test_df = jobs_test.unionAll(zuck_test)
print '[ INFO ] Number of Training Records: ' + str(test_df.count())

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
pipe = Pipeline(stages=[featurizer, lr])
pipe_model = pipe.fit(train_df)

predictions = pipe_model.transform(test_df)

predictions.select("filePath", "prediction").show(10,False)

predictionAndLabels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

#ZEND
