

from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id, col, expr, when, concat, lit
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator




#######################################################################################
#
#   Read in Data
#
#######################################################################################

mm_season = spark.read.load("/RegularSeasonDetailedResults.csv", format="csv", header=True)
#mm_season.createOrReplaceTempView('mm_season_sql')

mm_teams = spark.read.load("/Teams.csv", format="csv", header=True)
#mm_teams.createOrReplaceTempView('mm_teams_sql')

mm_tournament = spark.read.load("/TourneyDetailedResults.csv", format="csv", header=True)
#mm_tournament.createOrReplaceTempView('mm_tournament_sql')


mm_season.count()
mm_teams.count()
mm_tournament.count()



#######################################################################################
#
#   Combine Data
#
#######################################################################################

#mm_tournament = mm_tournament.withColumn("id", 900000 + monotonically_increasing_id())

alldata = mm_season.unionAll(mm_tournament)

alldata.show()
alldata.count()
mm_season.count()
mm_tournament.count()



#######################################################################################
#
#   Clean Up and Transform Data
#
#######################################################################################

# Set the lower ID as TeamA (as per Kaggle: https://www.kaggle.com/c/march-machine-learning-mania-2017#evaluation)
# TeamB will be the larger Team ID.
# Cast as desired Type

alldata_transformed = alldata                                                                           \
        .withColumn("Season", alldata["Season"].cast(IntegerType()))                                    \
        .withColumn("Daynum", alldata["Daynum"].cast(IntegerType()))                                    \
        .withColumn("TeamA",          expr("""IF(Wteam < Lteam, Wteam, Lteam)"""))                      \
        .withColumn("TeamA_Score",    expr("""IF(Wteam < Lteam, Wscore, Lscore)"""))                    \
        .withColumn("TeamA_OffReb",   expr("""IF(Wteam < Lteam, Wor, Lor)""").cast(IntegerType()))      \
        .withColumn("TeamA_DefReb",   expr("""IF(Wteam < Lteam, Wdr, Ldr)""").cast(IntegerType()))      \
        .withColumn("TeamA_Turnover", expr("""IF(Wteam < Lteam, Wto, Lto)""").cast(IntegerType()))      \
        .withColumn("TeamA_Steals",   expr("""IF(Wteam < Lteam, Wstl, Lstl)""").cast(IntegerType()))    \
        .withColumn("TeamA_Blocks",   expr("""IF(Wteam < Lteam, Wblk, Lblk)""").cast(IntegerType()))    \
        .withColumn("TeamA_Fouls",    expr("""IF(Wteam < Lteam, Wpf, Lpf)""").cast(IntegerType()))      \
                                                                                                        \
        .withColumn("TeamB",          expr("""IF(Wteam < Lteam, Lteam, Wteam)"""))                      \
        .withColumn("TeamB_Score",    expr("""IF(Wteam < Lteam, Lscore, Wscore)"""))                    \
        .withColumn("TeamB_OffReb",   expr("""IF(Wteam < Lteam, Lor, Wor)""").cast(IntegerType()))      \
        .withColumn("TeamB_DefReb",   expr("""IF(Wteam < Lteam, Ldr, Wdr)""").cast(IntegerType()))      \
        .withColumn("TeamB_Turnover", expr("""IF(Wteam < Lteam, Lto, Wto)""").cast(IntegerType()))      \
        .withColumn("TeamB_Steals",   expr("""IF(Wteam < Lteam, Lstl, Wstl)""").cast(IntegerType()))    \
        .withColumn("TeamB_Blocks",   expr("""IF(Wteam < Lteam, Lblk, Wblk)""").cast(IntegerType()))    \
        .withColumn("TeamB_Fouls",    expr("""IF(Wteam < Lteam, Lpf, Wpf)""").cast(IntegerType()))      \
                                                                                                        \
        .withColumn("WinFlag", expr("""IF(int(TeamA_Score) > int(TeamB_Score), 1, 0)"""))               \
        .withColumn("WinRatio", expr(""" round(TeamA_Score/float(TeamB_Score),2) """))                  \
        .withColumn("Matchup", concat("TeamA",lit("_"),"TeamB"))                                        \
        .select('season','daynum','WinFlag','WinRatio',                                                 \
                'TeamA','TeamB',                                                                        \
                'TeamA_OffReb','TeamB_OffReb',                                                          \
                'TeamA_DefReb','TeamB_DefReb',                                                          \
                'TeamA_Turnover','TeamB_Turnover',                                                      \
                'TeamA_Steals','TeamB_Steals',                                                          \
                'TeamA_Blocks','TeamB_Blocks',                                                          \
                'TeamA_Fouls','TeamB_Fouls')

alldata_transformed.show()


# For modeling purposes, I should take the average stats per team by season (or other interval). 
# To keep things simple, I'll make predictions based on individual games and generalize to a season average.
#test = alldata_transformed.groupBy('season',).agg({"TeamA_OffReb": "avg", "TeamA_DefReb": "avg"}).show()


# Use StringIndexer to convert all string inputs into Indexed Values (this converts multiple columns)
'''
>>>> Not needed with the current data structure

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(alldata) for column in ["TeamA","TeamB"] ]

pipeline = Pipeline(stages=indexers)
alldata_out = pipeline.fit(alldata).transform(alldata)

alldata_out.show()
'''


# Generate Features Vector and Label
va = VectorAssembler(inputCols=['season', 'daynum', 'TeamA_OffReb', 'TeamB_OffReb', 'TeamA_DefReb', 'TeamB_DefReb', 'TeamA_Turnover', 'TeamB_Turnover', 'TeamA_Steals', 'TeamB_Steals', 'TeamA_Blocks', 'TeamB_Blocks', 'TeamA_Fouls', 'TeamB_Fouls'], outputCol="features")
alldata_out_with_features = va.transform(alldata_transformed) \
                              .withColumn("label", col("WinFlag"))

alldata_out_with_features.show()



#######################################################################################
#
#   Define TRAINING and TESTING dataframes
#
#######################################################################################

# Random Spliting
#training, testing = alldata_out_with_features.randomSplit([0.8, 0.2])


# Holdout 2016 data for testing - train on all other seasons
training = alldata_out_with_features.filter('season != 2016')
testing  = alldata_out_with_features.filter('season == 2016')


alldata_out_with_features.count()
training.count()
testing.count()


#######################################################################################
#
#   Gradient Boost Model
#
#######################################################################################

#gbt = GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)
gbt = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", maxDepth=5, maxBins=32, maxIter=20, seed=12345)

gbtmodel = gbt.fit(training)

# Make predictions.
predictions = gbtmodel.transform(testing)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(10)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = " + str(accuracy))

# Accuracy = 0.862582781457



join_pred_with_team  = predictions.join(mm_teams, predictions.TeamA==mm_teams.Team_Id, "left")   \
                .drop("Team_Id")                                            \
                .withColumnRenamed("Team_Name", "TeamA_Name")

join_pred_with_team2 = join_pred_with_team.join(mm_teams, join_pred_with_team.TeamB==mm_teams.Team_Id, "left")   \
                .drop("Team_Id")                                            \
                .withColumnRenamed("Team_Name", "TeamB_Name")


join_pred_with_team2.where(                                                                 \
    join_pred_with_team2["TeamA_Name"].isin({"NC State", "North Carolina", "Duke"}) |       \
    join_pred_with_team2["TeamB_Name"].isin({"NC State", "North Carolina", "Duke"}))        \
    .select("TeamA_Name","TeamB_Name","label","prediction")                                 \
    .show(25,False)


#######################################################################################
#
#   Merge Predictions with Team Names
#
#######################################################################################

results = predictions.join(mm_teams, predictions.TeamA == mm_teams.Team_Id, 'left_outer')   \
                     .withColumnRenamed("Team_Name", "TeamA_Name").drop("Team_Id")          \
                     .join(mm_teams, predictions.TeamB == mm_teams.Team_Id, 'left_outer')   \
                     .withColumnRenamed("Team_Name", "TeamB_Name").drop("Team_Id")   




#ZEND
