
/*

This is a simple Spark (scala) example that shows how to load ESRI shapefile data into a Spark dataframe.
Reference: https://github.com/harsha2010/magellan
Test on Spark 2.2.0.2.6.4.0-91 and Scala 2.11.8

Setup Instructions: 

Step 1: Download test data:
    Test Data can be downloaded from http://www.arcgis.com/home/item.html?id=3b2a461c2c7848899b7b4cbfa9ebdb67
    This San Francisco Neighboorhoods dataset is in ESRI Shapefile format.

Step 2: Unzip
    unzip planning_neighborhoods.zip 

Step 3: Copy into HDFS
    rm planning_neighborhoods.zip 
    hadoop fs -mkdir /tmp/zesri
    hadoop fs -put planning_neighborhoods* /tmp/zesri
    hadoop fs -ls /tmp/zesri
    #Files in HDFS should look like this:
    #-rw-r--r--   3 hdfs hdfs       1028 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.dbf
    #-rw-r--r--   3 hdfs hdfs        567 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.prj
    #-rw-r--r--   3 hdfs hdfs        516 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.sbn
    #-rw-r--r--   3 hdfs hdfs        164 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.sbx
    #-rw-r--r--   3 hdfs hdfs     214576 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.shp
    #-rw-r--r--   3 hdfs hdfs      21958 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.shp.xml
    #-rw-r--r--   3 hdfs hdfs        396 2018-05-13 17:59 /tmp/zesri/planning_neighborhoods.shx

Step 4: Launch Spark
    /usr/hdp/current/spark2-client/bin/spark-shell --packages harsha2010:magellan:1.0.5-s_2.11

*/


import magellan.{Point, Polygon}
import org.apache.spark.sql.magellan.dsl.expressions._
import org.apache.spark.sql.types._

val df = spark.read.
            format("magellan").
            load("hdfs:///tmp/zesri")

df.show(20,false)
/* Output:
+-----+--------+-------------------------+--------------------------------------------+-----+
|point|polyline|polygon                  |metadata                                    |valid|
+-----+--------+-------------------------+--------------------------------------------+-----+
|null |null    |magellan.Polygon@7eb79e26|Map(neighborho -> Twin Peaks               )|true |
|null |null    |magellan.Polygon@eaae9fa3|Map(neighborho -> Pacific Heights          )|true |
|null |null    |magellan.Polygon@38da22ef|Map(neighborho -> Visitacion Valley        )|true |
|null |null    |magellan.Polygon@a6c2941d|Map(neighborho -> Potrero Hill             )|true |
|null |null    |magellan.Polygon@8929845a|Map(neighborho -> Crocker Amazon           )|true |
|null |null    |magellan.Polygon@96c26a08|Map(neighborho -> Outer Mission            )|true |
|null |null    |magellan.Polygon@615ecf6b|Map(neighborho -> Bayview                  )|true |
|null |null    |magellan.Polygon@e278fb12|Map(neighborho -> Lakeshore                )|true |
|null |null    |magellan.Polygon@81e1db25|Map(neighborho -> Russian Hill             )|true |
|null |null    |magellan.Polygon@b3c5c972|Map(neighborho -> Golden Gate Park         )|true |
|null |null    |magellan.Polygon@9cecca22|Map(neighborho -> Outer Sunset             )|true |
|null |null    |magellan.Polygon@1097d38 |Map(neighborho -> Inner Sunset             )|true |
|null |null    |magellan.Polygon@f3f4a823|Map(neighborho -> Excelsior                )|true |
|null |null    |magellan.Polygon@2c97aa67|Map(neighborho -> Outer Richmond           )|true |
|null |null    |magellan.Polygon@2116c80e|Map(neighborho -> Parkside                 )|true |
|null |null    |magellan.Polygon@bf9514fb|Map(neighborho -> Bernal Heights           )|true |
|null |null    |magellan.Polygon@33b17ef3|Map(neighborho -> Noe Valley               )|true |
|null |null    |magellan.Polygon@4e6eb334|Map(neighborho -> Presidio                 )|true |
|null |null    |magellan.Polygon@6864c27 |Map(neighborho -> Nob Hill                 )|true |
|null |null    |magellan.Polygon@e297c7e8|Map(neighborho -> Financial District       )|true |
+-----+--------+-------------------------+--------------------------------------------+-----+
only showing top 20 rows
*/

val neighborhoods = df.select(col("metadata")("neighborho"))

neighborhoods.show(20,false)
/* Ouput:
+-------------------------+
|metadata[neighborho]     |
+-------------------------+
|Twin Peaks               |
|Pacific Heights          |
|Visitacion Valley        |
|Potrero Hill             |
|Crocker Amazon           |
|Outer Mission            |
|Bayview                  |
|Lakeshore                |
|Russian Hill             |
|Golden Gate Park         |
|Outer Sunset             |
|Inner Sunset             |
|Excelsior                |
|Outer Richmond           |
|Parkside                 |
|Bernal Heights           |
|Noe Valley               |
|Presidio                 |
|Nob Hill                 |
|Financial District       |
+-------------------------+
*/


//ZEND
