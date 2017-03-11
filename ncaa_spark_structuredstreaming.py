

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col


############################################################################################################
#
#   Usage:
#   /spark/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.0.2
#
#   Kafka Producer:  
#        echo "DZ Kafka Event at $(date)" | /kafka/bin/kafka-console-producer.sh --broker-list spark_210:9092 --topic dztopic1 > /dev/null
#   Kafka Consumer:    
#       /kafka/bin/kafka-console-consumer.sh --zookeeper localhost:2181 --topic dztopic1 --from-beginning
#
#   Tested on:
#       Spark 2.0.2
#       Kafka 0.10.2
#       Python version 2.7.5 
#
############################################################################################################


#######################################################################################
#
#   Initialize Streaming Data Feed
#
#######################################################################################

'''
# Create DataFrame representing the stream of input lines from connection to localhost:9999
events = spark\
   .readStream\
   .format('socket')\
   .option('host', 'localhost')\
   .option('port', 9999)\
   .load()
'''


# Consume Kafka Topic
events = spark\
  .readStream\
  .format("kafka")\
  .option("kafka.bootstrap.servers", "localhost:9092")\
  .option("subscribe", "dztopic1")\
  .load()


#######################################################################################
#
#   Data Processing / Transformation on Streaming DF
#
#######################################################################################

#events.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

#events = spark.createDataFrame([('100001|1985|20|1228|81|1328|64|N|0',),('100002|1985|25|1106|77|1354|70|H|0',)],['value'])
events2 = events.withColumn('uid', split(events['value'],'\\|')[0] )    \
        .withColumn('season', split(events['value'],'\\|')[1] )         \
        .withColumn('daynum', split(events['value'],'\\|')[2] )         \
        .withColumn('wteam', split(events['value'],'\\|')[3] )          \
        .withColumn('wscore', split(events['value'],'\\|')[4] )         \
        .withColumn('lteam', split(events['value'],'\\|')[5] )          \
        .withColumn('lscore', split(events['value'],'\\|')[6] )         \
        .withColumn('wloc', split(events['value'],'\\|')[7] )           \
        .withColumn('ot', split(events['value'],'\\|')[8] )             \
        .withColumn('score_diff', col('wscore') - col('lscore') )



#######################################################################################
#
#   Load Static DF
#
#######################################################################################

staticDf = spark.read.load("/Teams.csv", format="csv", header=True)



#######################################################################################
#
#   Join Streaming DF with Static DF
#
#######################################################################################

dfjoin  = events2.join(staticDf, events2.wteam==staticDf.Team_Id, "left")   \
                .drop("Team_Id")                                            \
                .withColumnRenamed("Team_Name", "WTeam_Name")

dfjoin2 = dfjoin.join(staticDf,  dfjoin.lteam==staticDf.Team_Id, "left")    \
                .drop("Team_Id")                                            \
                .withColumnRenamed("Team_Name", "LTeam_Name")



#######################################################################################
#
#   Aggregations
#
#######################################################################################

# Aggregate by Winning Team
#teamCount = dfjoin2.groupBy('WTeam_Name','LTeam_Name')

# Aggregate by WTeam and LTeam, filtered for NC State, UNC, and Duke
teamCount = dfjoin2.where(                                                      \
    dfjoin2["WTeam_Name"].isin({"NC State", "North Carolina", "Duke"}) |        \
    dfjoin2["LTeam_Name"].isin({"NC State", "North Carolina", "Duke"}))         \
    .groupBy('WTeam_Name','LTeam_Name')                                         \
    .agg({"score_diff":"mean"})




#######################################################################################
#
#   Issue Structured Streaming Query
#
#######################################################################################

# Start Query (to console)
'''
query = teamCount\
    .writeStream\
    .outputMode('complete')\
    .format('console')\
    .start()
'''


# Start Query (to in-memory table)
query2 = teamCount\
    .writeStream\
    .format("memory")\
    .queryName("aggregates")\
    .outputMode("complete")\
    .start()


#######################################################################################
#
#   Issue Query against In-Memory / Realtime Streaming Table
#
#######################################################################################

# spark.sql("select * from aggregates order by count desc").show() 
spark.sql("select * from aggregates order by `avg(score_diff)` desc").show(20, False) 




#######################################################################################
#
#   Stop Structured Streaming Query
#
#######################################################################################

#query.awaitTermination()
#query.stop()
#query2.stop()


#ZEND
