
###########################################################################################
#
#   SparkR Code
#
#   Tested on Spark 2.0.0
#   http://spark.apache.org/docs/2.0.0/sparkr.html
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

head(rawdata)
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


###########################################################################################
#
#   Aggregations
#
###########################################################################################

# Number of reviews by Airline
head(summarize(groupBy(transformed, transformed$airline), number_of_reviews = count(transformed$airline)))


# Average Rating by Airline:
head(summarize(groupBy(transformed, transformed$airline), average_rating = mean(transformed$rating)))


# Average Rating by Airline and Cabin Type:
head(summarize(groupBy(transformed, transformed$airline, transformed$cabin), average_rating = mean(transformed$rating)))


# Number of Categories by "Airline"
head(summarize(groupBy(transformed, transformed$airline), number_of_reviews = count(transformed$id)))


# Number of Categories by "Location"
head(summarize(groupBy(transformed, transformed$location), number_of_reviews = count(transformed$id)))


# Number of Categories by "Cabin"
head(summarize(groupBy(transformed, transformed$cabin), number_of_reviews = count(transformed$id)))



###########################################################################################
#
#   SQL Operations
#
###########################################################################################

createOrReplaceTempView(transformed, "transformed_sql")

# Calculate the Average Rating by Airline, order by descending avg_rating
head(sql("
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

head(trainingDF)

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
head(sql("
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

# Fit a generalized linear model of family "binomial" with spark.glm
binomialGLM <- spark.glm(trainingDF, recommended ~ airline + cabin + year + month + value, family = "binomial")

# Model summary
summary(binomialGLM)

# Prediction
binomialPredictions <- predict(binomialGLM, testingDF)
showDF(binomialPredictions)




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
