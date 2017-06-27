
#!/bin/python
import urllib, sys, json
 
data = json.loads(urllib.urlopen("http://"+sys.argv[1]+":18080/api/v1/applications?status=running").read())
 
dyna = []
nondyna = []
 
#The History REST API for environments is not until Spark 2. So mining the html pages.
#Is not HA Safe, is not YARN App restart safe will only mine the first app attempts
for app in data:
    env = urllib.urlopen("http://"+sys.argv[1]+":18080/history/"+app['id']+"/1/environment").read()
    if "spark.executor.instances" not in env:
        dyna.append(app['id'])
    else:
        nondyna.append(app['id'])
 
print dyna
print nondyna

#Code credits Joe Niemiec
#ZEND
