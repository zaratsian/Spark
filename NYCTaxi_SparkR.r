

#######################################################################################
#
#   Load Data
#
#######################################################################################

rawdata <- read.df("/nyc_taxi_data.csv","csv", header = "true", inferSchema = "true", na.strings = "NA")

#head(rawdata)
showDF(rawdata)
schema(rawdata)
count(rawdata)

createOrReplaceTempView(rawdata, "rawdata_sql")


#######################################################################################
#
#   Explore Data
#
#######################################################################################

# How many unique cars are there (based on vehicle_id)
count(summarize(groupBy(rawdata, rawdata$vehicle_id), number_of_vehicles = count(rawdata$vehicle_id)))


# Describe / find basic stats for numerical data
showDF(describe(rawdata,"trip_distance","passenger_count","payment_amount","tip_amount"))


# Option 1 - What are my top earning cars
showDF(summarize(groupBy(rawdata, rawdata$vehicle_id), total_payment = sum(rawdata$payment_amount)) )


# Option 2 - What are my top earning cars
createOrReplaceTempView(rawdata, "rawdata_sql")
showDF(sql("SELECT vehicle_id, sum(payment_amount) as sum FROM rawdata_sql group by vehicle_id order by sum desc"))


# What is the average distance and average payment by car/vehicle
showDF(sql("SELECT vehicle_id, mean(trip_distance) as avg_distance, mean(payment_amount) as avg_payment FROM rawdata_sql group by vehicle_id order by avg_distance desc"))


# Covariance
cov(rawdata, 'payment_amount', 'fare_amount')
cov(rawdata, 'payment_amount', 'fare_amount')

# Correlation
corr(rawdata,"payment_amount", "fare_amount")
corr(rawdata,"tip_amount", "fare_amount")

# CrossTab
#showDF(crosstab(rawdata, "item1", "item2"))


#######################################################################################
#
#   Transformations
#
#######################################################################################

transformed1 <- selectExpr(rawdata, 
                            "vehicle_id",
                            "pickup_datetime",
                            "pickup_latitude",
                            "pickup_longitude",
                            "trip_distance",
                            "passenger_count",
                            "dropoff_datetime",
                            "dropoff_latitude",
                            "dropoff_longitude",
                            "fare_amount",
                            "tolls_amount",
                            "taxes_amount",
                            "tip_amount",
                            "payment_amount",
                            "cast(split(split(pickup_datetime,' ')[0],'/')[1] as int) pickup_day",
                            "cast(split(split(pickup_datetime,' ')[1],':')[0] as int) pickup_hour"
                            )

schema(transformed1)
#showDF(transformed1)


#######################################################################################
#
#   Model Prep
#
#######################################################################################

# Split into Training and Testing DFs
df_training_testing <- randomSplit(transformed1, weights=c(0.8, 0.2), seed=12345)

trainingDF <- df_training_testing[[1]]
testingDF  <- df_training_testing[[2]]

count(trainingDF)
count(testingDF)

#showDF(trainingDF)


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

gaussianGLM <- spark.glm(trainingDF, trip_distance ~ passenger_count + fare_amount + tolls_amount + tip_amount, family = "gaussian")

# Model summary
summary(gaussianGLM)

# Prediction
gaussianPredictions <- predict(gaussianGLM, testingDF)
showDF(select(gaussianPredictions, "label","prediction")




#ZEND