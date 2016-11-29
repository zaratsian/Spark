###############################################################################################################
#
#   Usage:
#
#   Download Data: ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz
#
###############################################################################################################

import re
import datetime
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, lag
from pyspark.sql.functions import udf

##################################################################################
#
#   Import Logs
#
##################################################################################

# docker cp ~/Downloads/NASA_access_log_Jul95 zeppelin:/.
logs = sc.textFile("/NASA_access_log_Jul95")
# HOST, TIMESTAMP (DAY MON DD HH:MM:SS YYYY, timezone is -0400), REQUEST, HTTP REPLY CODE, BYTES IN REPLY

##################################################################################
#
#   Clean & Parse Logs
#
##################################################################################

records = logs.map(lambda x: x.split(" "))

def cleanrecord(record):
    try:
        timestamp = datetime.datetime.strptime(record[3].replace("[",""), '%d/%b/%Y:%H:%M:%S')
        out = ( str(record[0]),timestamp,str(record[5]).replace('"',"").strip(),str(record[6]).strip(),record[8].strip(),int(record[9]) )
    except:
        out = ('',datetime.datetime.now(),'','','',0)
    
    return out

recordsDF = records.map(cleanrecord).toDF(("host","datetime","type","request","status_code","bytes"))

recordsDF.show(10,False)

'''
+--------------------+---------------------+----+-----------------------------------------------+-----------+-----+
|host                |datetime             |type|request                                        |status_code|bytes|
+--------------------+---------------------+----+-----------------------------------------------+-----------+-----+
|199.72.81.55        |1995-07-01 00:00:01.0|GET |/history/apollo/                               |200        |6245 |
|unicomp6.unicomp.net|1995-07-01 00:00:06.0|GET |/shuttle/countdown/                            |200        |3985 |
|199.120.110.21      |1995-07-01 00:00:09.0|GET |/shuttle/missions/sts-73/mission-sts-73.html   |200        |4085 |
|burger.letters.com  |1995-07-01 00:00:11.0|GET |/shuttle/countdown/liftoff.html                |304        |0    |
|199.120.110.21      |1995-07-01 00:00:11.0|GET |/shuttle/missions/sts-73/sts-73-patch-small.gif|200        |4179 |
|burger.letters.com  |1995-07-01 00:00:12.0|GET |/images/NASA-logosmall.gif                     |304        |0    |
|burger.letters.com  |1995-07-01 00:00:12.0|GET |/shuttle/countdown/video/livevideo.gif         |200        |0    |
|205.212.115.106     |1995-07-01 00:00:12.0|GET |/shuttle/countdown/countdown.html              |200        |3985 |
|d104.aa.net         |1995-07-01 00:00:13.0|GET |/shuttle/countdown/                            |200        |3985 |
|129.94.144.152      |1995-07-01 00:00:13.0|GET |/                                              |200        |7074 |
+--------------------+---------------------+----+-----------------------------------------------+-----------+-----+
'''

##################################################################################
#
#   Partition by HOST
#   Sort by DATETIME
#
#   Capture previous datetime of session.
#   This will be used to calculate time since last visit / session duration.
#
##################################################################################

wSpec = Window.partitionBy("host").orderBy("datetime")
sessions1 = recordsDF.filter(recordsDF.bytes>=50000).withColumn("previous_time", lag(recordsDF.datetime, 1).over(wSpec) )
sessions1.show(10,False)

'''
+--------------+---------------------+----+------------------------------------------------------+-----------+------+---------------------+
|host          |datetime             |type|request                                               |status_code|bytes |previous_time        |
+--------------+---------------------+----+------------------------------------------------------+-----------+------+---------------------+
|128.158.26.109|1995-07-05 13:04:31.0|GET |/shuttle/missions/sts-71/images/KSC-95EC-0917.jpg     |200        |52491 |null                 |
|128.158.26.109|1995-07-05 13:36:34.0|GET |/shuttle/missions/sts-71/images/KSC-95EC-0912.jpg     |200        |66202 |1995-07-05 13:04:31.0|
|128.158.26.109|1995-07-05 13:51:24.0|GET |/shuttle/technology/sts-newsref/stsref-toc.html       |200        |84907 |1995-07-05 13:36:34.0|
|128.158.26.109|1995-07-05 13:52:45.0|GET |/shuttle/technology/sts-newsref/sts_asm.html          |200        |71656 |1995-07-05 13:51:24.0|
|128.158.26.109|1995-07-05 13:53:02.0|GET |/shuttle/technology/images/srb_mod_compare_3-small.gif|200        |55666 |1995-07-05 13:52:45.0|
|128.158.26.109|1995-07-05 14:37:37.0|GET |/shuttle/technology/images/srb_mod_compare_3.jpg      |200        |258334|1995-07-05 13:53:02.0|
|128.158.44.230|1995-07-06 11:54:06.0|GET |/shuttle/technology/sts-newsref/spacelab.html         |200        |104916|null                 |
|128.158.44.230|1995-07-10 14:56:37.0|GET |/shuttle/technology/sts-newsref/spacelab.html         |200        |104916|1995-07-06 11:54:06.0|
|128.158.44.230|1995-07-17 12:06:20.0|GET |/shuttle/technology/sts-newsref/spacelab.html         |200        |104914|1995-07-10 14:56:37.0|
|128.158.44.230|1995-07-20 14:55:02.0|GET |/shuttle/technology/sts-newsref/stsref-toc.html       |200        |84905 |1995-07-17 12:06:20.0|
+--------------+---------------------+----+------------------------------------------------------+-----------+------+---------------------+
'''

def time_delta(x,y):
    try:
        #start = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        #end   = datetime.datetime.strptime(y, '%Y-%m-%d %H:%M:%S.%f')
        delta = int((y-x).total_seconds())
    except:
        delta = 0
    return delta

# register as a UDF
f = udf(time_delta, IntegerType())

sessions1.withColumn('duration', f(sessions1.previous_time, sessions1.datetime)).show(10,False)


sessions2 = sessions1.withColumn("elapsedtime", sessions1.datetime - sessions1.previous_time)
recordsDF.filter(recordsDF.bytes>=50000).withColumn( "rank", rank().over(wSpec) ).show(10,False)


#ZEND
