
###########################################################################################
#
#   SparkR Code
#
#   Tested on Hortonworks HDP 2.5
#   Spark 2.0.0
#   http://spark.apache.org/docs/2.0.0/sparkr.html
#
#   In Zeppelin, make sure to start Livy Server (start up on port 8998)
#   su - livy
#   /usr/hdp/current/livy-client/bin/livy-server start
#
###########################################################################################


###########################################################################################
#
#   Import Data
#
###########################################################################################

rawdata <- read.df("hdfs:/demo/data/airlines/airlines.csv","csv", header = "true", inferSchema = "false", na.strings = "NA")

head(rawdata)
schema(rawdata)
count(rawdata)

###########################################################################################
#
#   Data Processing / Transformations
#
###########################################################################################

# Keep only certain records (drop the column which contains airline reviews, "review")
rawdata$review <- NULL

showDF(rawdata)
schema(rawdata)
count(rawdata)

#     id         airline       date location rating    cabin value recommended
#1 10001 Delta Air Lines 2015-06-21 Thailand      7  Economy     4         YES
#2 10002 Delta Air Lines 2015-06-19      USA      0  Economy     2          NO
#3 10003 Delta Air Lines 2015-06-18      USA      0  Economy     1          NO
#4 10004 Delta Air Lines 2015-06-17      USA      9 Business     4         YES
#5 10005 Delta Air Lines 2015-06-17  Ecuador      7  Economy     3         YES
#6 10006 Delta Air Lines 2015-06-17      USA      9 Business     5         YES


# Derive Year, Month, and Day from the date variable.
transformed <- selectExpr(rawdata, 
                            "id",
                            "date",
                            "airline",
                            "location",
                            "cast(rating as int) rating",
                            "cabin",
                            "value",
                            "recommended",
                            "substr(date, 1, 4) as year",
                            "substr(date, 6, 2) as month",
                            "substr(date, 9, 2) as day"
                            )

schema(transformed)


# Filter results to show only Delta data
# Where Airline = Delta and Location = USA
head(filter(transformed, transformed$airline == "Delta Air Lines" & transformed$location == "USA"))


showDF(transformed)


###########################################################################################
#
#   Aggregations
#
###########################################################################################

# Number of reviews by Airline
showDF(summarize(groupBy(transformed, transformed$airline), number_of_reviews = count(transformed$airline)))


# Average Rating by Airline:
showDF(summarize(groupBy(transformed, transformed$airline), number_of_reviews = count(transformed$airline), average_rating = mean(transformed$rating)))


# Average Rating by Airline and Cabin Type:
showDF(summarize(groupBy(transformed, transformed$airline, transformed$cabin), average_rating = mean(transformed$rating)))


# Number of Categories by "Airline"
showDF(summarize(groupBy(transformed, transformed$airline), number_of_reviews = count(transformed$id)))


# Number of Categories by "Location"
showDF(summarize(groupBy(transformed, transformed$location), number_of_reviews = count(transformed$id)))


# Number of Categories by "Cabin"
showDF(summarize(groupBy(transformed, transformed$cabin), number_of_reviews = count(transformed$id)))



###########################################################################################
#
#   SQL Operations
#
###########################################################################################

createOrReplaceTempView(transformed, "transformed_sql")

# Calculate the Average Rating by Airline, order by descending avg_rating
showDF(sql("
    SELECT airline, count(*) as number_of_reviews, mean(rating) as avg_rating 
    FROM transformed_sql 
    group by airline 
    order by avg_rating desc"))



###########################################################################################
#
#   Applying custom function (UDF)
#
###########################################################################################

s <- structType(structField("id", "string"),
                structField("airline", "string"),
                structField("rating", "integer"), 
                structField("test", "integer"))

test <- dapply(transformed, function(x)
        { 
            temp <- x[5] + 500L
            test <- cbind(x[1], x[2], x[5], temp )
        },
        s)

head(test)



###########################################################################################
#
#   Modeling (NaiveBayes)
#
###########################################################################################

# Split into Training and Testing DFs
df_training_testing <- randomSplit(transformed, weights=c(0.8, 0.2), seed=12345)

trainingDF <- df_training_testing[[1]]
testingDF  <- df_training_testing[[2]]

count(trainingDF)
count(testingDF)

showDF(trainingDF)

nbmodel <- spark.naiveBayes(trainingDF, recommended ~ airline + cabin + year + month + value )


# Model summary
summary(nbmodel)

# Prediction
nbPredictions <- predict(nbmodel, testingDF)
showDF(nbPredictions, 25)

# Show accuracy matrix
accuracy1 <- selectExpr(
    summarize(groupBy(nbPredictions, nbPredictions$recommended, nbPredictions$prediction), number_of_reviews = count(transformed$id))
    ,
    "recommended",
    "prediction",
    "number_of_reviews",
    "IF (recommended==prediction, 1, 0) as accurate")

showDF(accuracy1)

#+-----------+----------+-----------------+--------+
#|recommended|prediction|number_of_reviews|accurate|
#+-----------+----------+-----------------+--------+
#|         NO|       YES|               27|       0|
#|        YES|       YES|               49|       1|
#|         NO|        NO|              110|       1|
#|        YES|        NO|               26|       0|
#+-----------+----------+-----------------+--------+

accuracy2 <- summarize(groupBy(accuracy1, accuracy1$accurate), count = sum(accuracy1$number_of_reviews))

showDF(accuracy2)

# Calculate Accuracy Score
createOrReplaceTempView(nbPredictions, "nbPredictions_sql")
showDF(sql("
    select (sum(*) / count(*)) as Accuracy_Score 
    from 
        (SELECT IF(recommended==prediction, 1, 0) as accuracy FROM nbPredictions_sql)
    "))

# Accuracy_Score
#           0.75



#############################
# Save and Load Model
#############################
#modelPath <- tempfile(pattern = "/tmp/ml", fileext = ".tmp")
#write.ml(nbmodel, modelPath)   
#nbmodel2 <- read.ml(modelPath)
#unlink(modelPath)




###########################################################################################
#
#   Modeling (GLM)
#   (1) Gaussian & (2) Binomial Predictions
#
###########################################################################################

# Family may include (https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html): 
#    binomial(link = "logit")
#    gaussian(link = "identity")
#    Gamma(link = "inverse")
#    inverse.gaussian(link = "1/mu^2")
#    poisson(link = "log")
#    quasi(link = "identity", variance = "constant")
#    quasibinomial(link = "logit")
#    quasipoisson(link = "log")

gaussianGLM <- spark.glm(trainingDF, rating ~ airline + cabin + year + month + value, family = "gaussian")

# Model summary
summary(gaussianGLM)

# Prediction
gaussianPredictions <- predict(gaussianGLM, testingDF)
showDF(gaussianPredictions)

#+-----+----------+---------------+-----------+------+-----------+-----+-----------+----+-----+---+-----+------------------+
#|   id|      date|        airline|   location|rating|      cabin|value|recommended|year|month|day|label|        prediction|
#+-----+----------+---------------+-----------+------+-----------+-----+-----------+----+-----+---+-----+------------------+
#|10005|2015-06-17|Delta Air Lines|    Ecuador|     7|    Economy|    3|        YES|2015|    6| 17|  7.0| 4.543003058474369|
#|10008|2015-06-14|Delta Air Lines|        USA|     0|    Economy|    1|         NO|2015|    6| 14|  0.0| 1.548625588686491|
#|10009|2015-06-13|Delta Air Lines|        USA|     4|   Business|    2|         NO|2015|    6| 13|  4.0| 5.146936019287523|
#|10016|2015-06-05|Delta Air Lines|        USA|     0|    Economy|    1|         NO|2015|    6|  5|  0.0| 1.548625588686491|
#|10017|2015-06-03|Delta Air Lines|     Canada|     9|    Economy|    4|        YES|2015|    6|  3|  9.0| 6.040191793368194|
#|10018|2015-06-02|Delta Air Lines|        USA|     9|    Economy|    5|        YES|2015|    6|  2|  9.0|  7.53738052826202|
#|10036|2015-05-05|Delta Air Lines|        USA|     5|    Economy|    3|         NO|2015|    5|  5|  5.0|4.5656995657975585|
#|10077|2015-03-12|Delta Air Lines|        USA|     5|    Economy|    3|         NO|2015|    3| 12|  5.0|  4.61109258044371|
#|10080|2015-03-03|Delta Air Lines|        USA|     0|First Class|    1|         NO|2015|    3|  3|  0.0| 2.878129135126983|
#|10081|2015-03-03|Delta Air Lines|    Germany|    10|   Business|    4|        YES|2015|    3|  3| 10.0| 8.209403011044742|
#|10089|2015-02-18|Delta Air Lines|        USA|     2|    Economy|    2|         NO|2015|    2| 18|  2.0|3.1366003528726196|
#|10094|2015-02-10|Delta Air Lines|    Ireland|     7|    Economy|    3|        YES|2015|    2| 10|  7.0| 4.633789087766672|
#|10102|2015-01-20|Delta Air Lines|        USA|     0|    Economy|    1|         NO|2015|    1| 20|  0.0|1.6621081253019838|
#+-----+----------+---------------+-----------+------+-----------+-----+-----------+----+-----+---+-----+------------------+



# Fit a generalized linear model of family "binomial" with spark.glm
binomialGLM <- spark.glm(trainingDF, recommended ~ airline + cabin + year + month + value, family = "binomial")

# Model summary
summary(binomialGLM)

# Prediction
binomialPredictions <- predict(binomialGLM, testingDF)
showDF(binomialPredictions)

#+-----+----------+---------------+-----------+------+-----------+-----+-----------+----+-----+---+-----+-------------------+
#|   id|      date|        airline|   location|rating|      cabin|value|recommended|year|month|day|label|         prediction|
#+-----+----------+---------------+-----------+------+-----------+-----+-----------+----+-----+---+-----+-------------------+
#|10005|2015-06-17|Delta Air Lines|    Ecuador|     7|    Economy|    3|        YES|2015|    6| 17|  1.0| 0.5647025067536203|
#|10008|2015-06-14|Delta Air Lines|        USA|     0|    Economy|    1|         NO|2015|    6| 14|  0.0|0.16484310555455592|
#|10009|2015-06-13|Delta Air Lines|        USA|     4|   Business|    2|         NO|2015|    6| 13|  0.0| 0.5491552805132383|
#|10016|2015-06-05|Delta Air Lines|        USA|     0|    Economy|    1|         NO|2015|    6|  5|  0.0|0.16484310555455592|
#|10017|2015-06-03|Delta Air Lines|     Canada|     9|    Economy|    4|        YES|2015|    6|  3|  1.0| 0.7688300479888639|
#|10018|2015-06-02|Delta Air Lines|        USA|     9|    Economy|    5|        YES|2015|    6|  2|  1.0| 0.8950282664177464|
#|10036|2015-05-05|Delta Air Lines|        USA|     5|    Economy|    3|         NO|2015|    5|  5|  0.0|  0.545022785906816|
#|10041|2015-05-04|Delta Air Lines|Switzerland|    10|    Economy|    5|        YES|2015|    5|  4|  1.0| 0.8873021365295578|
#|10043|2015-05-04|Delta Air Lines|        USA|     1|    Economy|    1|         NO|2015|    5|  4|  0.0| 0.1541632118831403|
#|10080|2015-03-03|Delta Air Lines|        USA|     0|First Class|    1|         NO|2015|    3|  3|  0.0| 0.1975240148562852|
#|10081|2015-03-03|Delta Air Lines|    Germany|    10|   Business|    4|        YES|2015|    3|  3|  1.0|  0.863077070741532|
#|10089|2015-02-18|Delta Air Lines|        USA|     2|    Economy|    2|         NO|2015|    2| 18|  0.0| 0.2689543087086207|
#|10094|2015-02-10|Delta Air Lines|    Ireland|     7|    Economy|    3|        YES|2015|    2| 10|  1.0|0.48538240625978296|
#|10102|2015-01-20|Delta Air Lines|        USA|     0|    Economy|    1|         NO|2015|    1| 20|  0.0| 0.1170082958954653|
#+-----+----------+---------------+-----------+------+-----------+-----+-----------+----+-----+---+-----+-------------------+



###########################################################################################
#
#   Modeling (KMeans Clustering) - Unsupervised
#
###########################################################################################

kmeansdata <- selectExpr(rawdata, 
                            "id",
                            "date",
                            "airline",
                            "location",
                            "cast(rating as int) as rating",
                            "cabin",
                            "cast(value as int) as value",
                            "recommended",
                            "cast(substr(date, 1, 4) as int) as year",
                            "cast(substr(date, 6, 2) as int) as month",
                            "cast(substr(date, 9, 2) as int) as day"
                            )

showDF(kmeansdata)
schema(kmeansdata)

# Split into Training and Testing DFs
df_training_testing <- randomSplit(kmeansdata, weights=c(0.8, 0.2), seed=12345)

trainingDF <- df_training_testing[[1]]
testingDF  <- df_training_testing[[2]]

count(trainingDF)
count(testingDF)

kmeansModel <- spark.kmeans(trainingDF, ~ rating + value + year + month + day,
                            k = 5)

# Model summary
summary(kmeansModel)

# Get fitted result from the k-means model
showDF(fitted(kmeansModel))

# Make predictions on holdout/test data
kmeansPredictions <- predict(kmeansModel, testingDF)
showDF(kmeansPredictions)




#ZEND
