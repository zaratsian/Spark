
# coding: utf-8

# ![CRISP-DM](https://raw.githubusercontent.com/zaratsian/Spark/master/nfl_banner2.png)
# 

# # Use Case:  Predicting NFL play

# ### Loading Libraries

# In[1]:

import datetime, time 
import re, random, sys
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, StringType, FloatType, LongType
from pyspark.sql.functions import struct, array, lit, monotonically_increasing_id, col, expr, when, concat, udf, split, size, lag, count, isnull
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor, LinearRegression, GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer, IndexToString
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator


# In[2]:

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
#sc = SparkContext()
sparkSession = SparkSession(sc).builder.getOrCreate()


# # 1. Data Ingestion

# ### Peeking into data

# In[4]:

get_ipython().system(u'curl -i -L "http://edwdemo0.field.hortonworks.com:50070/webhdfs/v1/data/NFLPlaybyPlay2015.csv?op=OPEN" | tail -n 5')


# ### Load Data from Remote HDP Cluster (from HDFS)

# In[5]:

rawdata = sparkSession.read.csv('hdfs://edwdemo0.field.hortonworks.com:8020/data/NFLPlaybyPlay2015.csv', header=True, inferSchema=True)

print '\nTotal Records: ' + str(rawdata.count()) + '\n'
for i in rawdata.dtypes: print i

rawdata = rawdata.select( [rawdata['`'+c+'`'].alias(c.replace('.','_')) for c in rawdata.columns] )


# # 2. Data Wrangling: Cleaning , Transformations, Enrichment

# ## Data Cleaning & Transformations

# In[6]:

columns_to_keep =   [   
                    "Date", "GameID", "Drive", "qtr", "down", "time", "TimeUnder", "TimeSecs", 
                    "PlayTimeDiff", "yrdline100", "ydstogo", "ydsnet", "FirstDown", "posteam", 
                    "DefensiveTeam", "Yards_Gained", "Touchdown", "PlayType", "PassLength", 
                    "PassLocation", "RunLocation",
                    #"Passer", "Rusher", "InterceptionThrown", "Season"
                    "PosTeamScore", "DefTeamScore"
                    ]

# Filter columns (keep)
nfldata = rawdata.select(columns_to_keep)

# Drop rows with NAa:
nfldata = nfldata.filter(nfldata.down != 'NA')

# approxQuantile
nfldata.approxQuantile(col='Yards_Gained', probabilities=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], relativeError=0.05)

# Filter target variable (Yards_Gained) to remove outliers
nfldata = nfldata.filter( (col('Yards_Gained') <= 20 ) & (col('Yards_Gained') >= -5 ) )
nfldata.approxQuantile(col='Yards_Gained', probabilities=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], relativeError=0.05)

numeric_columns     = [c[0] for c in nfldata.dtypes if c[1] not in ['string','timestamp']]
categorical_columns = [c[0] for c in nfldata.dtypes if c[1] in ['string']]
datetime_columns    = [c[0] for c in nfldata.dtypes if c[1] in ['timestamp']]


# ## Data Enrichment & Additional Transformations

# In[7]:


nfldata2 = nfldata.withColumn("Date",            col("Date"))                                           .withColumn("GameID",       col("GameID").cast("int"))                              .withColumn("Drive",        col("Drive").cast("int"))                               .withColumn("qtr",          col("qtr").cast("int"))                                 .withColumn("down",         col("down").cast("int"))                                .withColumn("time",         col("time").cast("string"))                             .withColumn("TimeUnder",    col("TimeUnder").cast("int"))                           .withColumn("TimeSecs",     col("TimeSecs").cast("int"))                            .withColumn("PlayTimeDiff", col("PlayTimeDiff").cast("int"))                        .withColumn("yrdline100",   col("yrdline100").cast("int"))                          .withColumn("ydstogo",      col("ydstogo").cast("int"))                             .withColumn("ydsnet",       col("ydsnet").cast("int"))                              .withColumn("FirstDown",    col("FirstDown").cast("int"))                           .withColumn("posteam",      col("posteam").cast("string"))                          .withColumn("DefensiveTeam",col("DefensiveTeam").cast("string"))                    .withColumn("Yards_Gained", col("Yards_Gained").cast("int"))                        .withColumn("Touchdown",    col("Touchdown").cast("int"))                           .withColumn("PlayType",     col("PlayType").cast("string"))                         .withColumn("PassLength",   col("PassLength").cast("string"))                       .withColumn("PassLocation", col("PassLocation").cast("string"))                     .withColumn("RunLocation",  col("RunLocation").cast("string"))                      .withColumn("PosTeamScore", col("PosTeamScore").cast("int"))                        .withColumn("DefTeamScore", col("DefTeamScore").cast("int")) 


numeric_columns     = [c[0] for c in nfldata2.dtypes if c[1] not in ['string','timestamp']]
categorical_columns = [c[0] for c in nfldata2.dtypes if c[1] in ['string']]
datetime_columns    = [c[0] for c in nfldata2.dtypes if c[1] in ['timestamp']]

'''
# Correlation
seen = []
for c1 in numeric_columns:
    for c2 in numeric_columns:
        correlation = round(nfldata2.stat.corr(c1, c2), 8)
        #if (correlation >= 0.90 or correlation <= 0.10) and (c1 != c2) and ((c1,c2) not in seen):
        if (correlation >= 0.70) and (c1 != c2) and ((c1,c2) not in seen):
            seen.append((c2,c1))
            print str(correlation) + '\tCorrelation for ' + str(c1) + ' and ' + str(c2)
'''

# Category Levels
[nfldata2.select(nfldata2[c]).groupBy(nfldata2[c]).count().show(5,False) for c in categorical_columns]


# ## Data Enrichment & Additional Transformations (Continued...)

# In[8]:

# Filter - Keep where Playtype in ['Run','Pass'] 
nfldata2 = nfldata2.filter( (nfldata2.PlayType=="Run") | (nfldata2.PlayType=="Pass") )

# Derive Date var(s)
nfldata2 = nfldata2.withColumn("month_day", concat(nfldata2["Date"].substr(6,2), nfldata2["Date"].substr(9,2)).cast("int") )

# Lag (Get previous PlayType)
w = Window().partitionBy('GameID','Drive').orderBy('GameID','Drive', col('TimeSecs').desc())
nfldata2 = nfldata2.withColumn("PlayType_lag", lag("PlayType").over(w) )                  .withColumn("PlayType_lag",  when( isnull('PlayType_lag'), 'FirstPlay').otherwise( col('PlayType_lag') ) )                  .orderBy('GameID','Drive', col('TimeSecs').desc())

# Print Results
#nfldata2.select(["GameID","Drive","qtr","down","TimeSecs","PlayType","PlayType_lag","yrdline100","posteam","month_day"]).show(50,False)

# Split into "Run" and "Pass" (I want to build two models)
nfldata2_run  = nfldata2.filter( col('PlayType')=='Run' )
nfldata2_pass = nfldata2.filter( col('PlayType')=='Pass' )

print "Total Number of Records:   " + str(nfldata2.count())
print "Number of Running Records: " + str(nfldata2_run.count())
print "Number of Passing Records: " + str(nfldata2_pass.count())


# # 3. Data Exploration

# In[9]:

# Inspecting RDDs
print(type(nfldata2))


# In[10]:

# generating pandas dataframe for visualizations
run_pd = nfldata2_run.toPandas()


# In[11]:

pass_pd= nfldata2_pass.toPandas()


# In[12]:

# viewing the run data
run_pd.head(5)


# In[13]:

# viewing the pass data
pass_pd.head(5)


# In[14]:

import seaborn as sns
import warnings
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# set plot size
fig_size=[0,0]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

# setting style
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')


# ### Correlation Matrix for features

# In[15]:

# Compute the correlation matrix
corr = run_pd.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})


# ### Visualizing multidimensional relationships
# 
# *exploring correlations between multidimensional data, when you'd like to plot all pairs of values against each other.*

# In[16]:

# histograms
run_pd.hist()
plt.show()


# In[17]:

# pair plot
sns.pairplot(run_pd, hue='Yards_Gained', size=2.5);
plt.show() 


# In[18]:

# Linear Regression Plot
lm=sns.regplot(x="PosTeamScore", y="qtr", data=run_pd)
plt.show()


# In[19]:

# bar charts
sns.barplot(y="down", x="FirstDown", data=run_pd)


# In[20]:

# bar chart 3 features
ax = sns.barplot(x="yrdline100", y="ydsnet", data=run_pd)


# In[21]:

# distribution plot
sns.distplot(run_pd["Drive"]);


# # Model Building

# ### Split into Train and Test

# In[22]:

training_run, testing_run   = nfldata2_run.randomSplit([0.8, 0.2], seed=12345)
training_pass, testing_pass = nfldata2_pass.randomSplit([0.8, 0.2], seed=12345)


# ### Building Model Pipeline

# In[23]:

# Prepare string variables so that they can be used by the decision tree algorithm
# StringIndexer encodes a string column of labels to a column of label indices
si1 = StringIndexer(inputCol="PlayType", outputCol="PlayType_index")
si2 = StringIndexer(inputCol="PlayType_lag", outputCol="PlayType_lag_index")
si3 = StringIndexer(inputCol="PassLength", outputCol="PassLength_index")
si4 = StringIndexer(inputCol="PassLocation", outputCol="PassLocation_index")
si5 = StringIndexer(inputCol="RunLocation", outputCol="RunLocation_index")

target   = 'Yards_Gained'
features = ['qtr','down','TimeSecs','yrdline100','ydstogo','ydsnet','month_day','PlayType_lag_index']

#encode the Label column: feature indexer
fi = StringIndexer(inputCol='Yards_Gained', outputCol='label').fit(training_run)

# Pipelines API requires that input variables are passed in  a vector
va  = VectorAssembler(inputCols=features, outputCol="features")


# In[24]:

# run the algorithm and build model taking the default settings
rfr = RandomForestRegressor(featuresCol="label", labelCol=target, predictionCol="prediction", maxDepth=5, maxBins=350, seed=12345)
gbr = GBTRegressor(featuresCol="features", labelCol=target, predictionCol="prediction", maxDepth=5, maxBins=350, seed=12345)

# Convert indexed labels back to original labels, label converter
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=fi.labels)


# ### Training the Model

# In[ ]:

# Build the machine learning pipeline
pipeline_run  = Pipeline(stages=[si2, fi, va, gbr, labelConverter])

# Build model. 
# The fitted model from a Pipeline is a PipelineModel, which consists of fitted models and transformers, corresponding to the pipeline stages.
model_run = pipeline_run.fit(training_run)

# store the predictions on training data on HDFS
#model_run.write().overwrite().save('hdfs://dzaratsian0.field.hortonworks.com:8020/models/nfl_model_run3')

#print(model_run.bestModel.stages[-2].featureImportances)


# ### Making predictions for model

# In[43]:

# Make predictions.
predictions = model_run.transform(testing_run)
# show the results
predictions.show(3)


# ### Generate results of classifier

# In[46]:

predictions=predictions.select(predictions["Yards_Gained"],predictions["predictedLabel"],predictions["prediction"])
type(predictions)


# In[47]:

predictions.show(5)


# ### Model Evaluation

# In[48]:

# Evaluate Results
evaluator = RegressionEvaluator(metricName="rmse", labelCol=target)  # rmse (default)|mse|r2|mae
RMSE = evaluator.evaluate(predictions)
print 'RMSE: ' + str(RMSE)

evaluator = RegressionEvaluator(metricName="mae", labelCol=target)  # rmse (default)|mse|r2|mae
MAE = evaluator.evaluate(predictions) # Mean Absolute Error
print 'MSE: ' + str(MAE)


# # Model Management: Save & Deploy

# In[49]:

from repository.mlrepositoryclient import MLRepositoryClient
from repository.mlrepositoryartifact import MLRepositoryArtifact


# In[50]:

service_path = 'https://internal-nginx-svc.ibm-private-cloud.svc.cluster.local:12443'
ml_repository_client = MLRepositoryClient()


# In[52]:

type(model_run)


# ### Create model artifact (abstraction layer)

# In[53]:

model_artifact = MLRepositoryArtifact(model_run, training_data=training_run, name="NFL Game Prediction")


# ### Save pipeline and model artifacts to in Machine Learning repository

# In[54]:

saved_model = ml_repository_client.models.save(model_artifact)


# ### Saved model properties

# In[55]:

print "modelType: " + saved_model.meta.prop("modelType")
print "creationTime: " + str(saved_model.meta.prop("creationTime"))
print "modelVersionHref: " + saved_model.meta.prop("modelVersionHref")
print "label: " + saved_model.meta.prop("label")


# In[66]:

tr=training_run.toPandas()


# In[71]:

list(tr.columns)


# In[68]:

# values for input
for i in tr.columns:
    print(i, run_pd[i][10])
'2015-09-10 00:00:00', 2015091000, 4, 2, 1,'14:30', 15, 2670, 30, 63, 10, 24, 0, 'NE', 'PIT', -3, 0, 'Run', 'NA', 'NA', 'left', 0, 0, 910, 'Run'



# # Model Testing: UI & API 

# ## UI Testing

# In[ ]:




# ## API Testing

# ### Retreiving  bearer token 

# In[73]:

get_ipython().system(u'curl -k -X GET https://172.26.222.224/v2/identity/token -H "username: dzaratsian" -H "password: BadPass#1"')


# ### Invoke model remotely

# !curl -i -k -X POST https://172.26.228.121/v2/scoring/online/4a5692ff-2bfc-42d4-a005-ebbd8ffea88a -d '{"fields": ["isCertified","paymentScheme","hoursDriven","milesDriven","latitude","longitude","isFoggy","isRainy","isWindy"], "records": [["N","miles",0.000000,0.000000,-94.590000,39.100000,0.000000,0.000000,0.000000]]}' -H "content-type: application/json" -H "authorization: Bearer <token>"

# In[ ]:

get_ipython().system(u'curl -i -k -X POST https://172.26.228.121/v2/scoring/online/2304b6e2-51aa-4e57-821d-35b565caf8cf -d \'{"fields":[\'Date\', \'GameID\', \'Drive\',\'qtr\',\'down\',\'time\',\'TimeUnder\',\'TimeSecs\',\'PlayTimeDiff\',\'yrdline100\', \'ydstogo\', \'ydsnet\', \'FirstDown\', \'posteam\', \'DefensiveTeam\', \'Yards_Gained\',\'Touchdown\',\'PlayType\', \'PassLength\', \'PassLocation\', \'RunLocation\', \'PosTeamScore\', \'DefTeamScore\', \'month_day\', \'PlayType_lag\']},"records":[[\'2015-09-10 00:00:00\', 2015091000, 4, 2, 1,\'14:30\', 15, 2670, 30, 63, 10, 24, 0, \'NE\', \'PIT\', -3, 0, \'Run\', \'NA\', \'NA\', \'left\', 0, 0, 910, \'Run\']]\'')
    
    
    

