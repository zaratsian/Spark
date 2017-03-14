
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

accuracy2 <- summarize(groupBy(accuracy1, accuracy1$accurate), count = sum(accuracy1$number_of_reviews))

showDF(accuracy2)

# Calculate Accuracy Score
createOrReplaceTempView(nbPredictions, "nbPredictions_sql")
head(sql("
    select (sum(*) / count(*)) as Accuracy_Score 
    from 
        (SELECT IF(recommended==prediction, 1, 0) as accuracy FROM nbPredictions_sql)
    "))



#ZEND
