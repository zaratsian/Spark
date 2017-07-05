
##########################################################################################
#
#   Spark Tuning Tool
#
#   Dan Zaratsian
#
#   Usage: spark_tuning_tool_ambari.py --ambari_url=http://dzaratsian-hdp1:8080/api/v1/clusters/dz_hdp/services/HDFS/components/DATANODE --username=admin --password=admin
#
##########################################################################################


import sys,re
import getopt
import requests
import json
import math


try:
    opts, args = getopt.getopt(sys.argv[1:], 'x', ['ambari_url=', 'username=', 'password='])
    
    ambari_url = [opt[1] for opt in opts if opt[0]=='--ambari_url'][0]
    username   = [opt[1] for opt in opts if opt[0]=='--username'][0]
    password   = [opt[1] for opt in opts if opt[0]=='--password'][0]
except:
    print '\n\n[ USAGE ] spark_tuning_tool_ambari.py --ambari_url=<url> --username=<string> --password=<string>\n\n'
    sys.exit(1)


node_count = []
node_cores = []
node_ram   = []


req1 = requests.get(ambari_url, auth=(username,password) )
if req1.status_code == 200:
    cluster_info = json.loads(req1.content)
    
    for host in cluster_info['host_components']:
        print '[ INFO ] Collection data from ' + str(host['href'])
        #req2 = requests.get('http://dzaratsian-hdp1:8080/api/v1/clusters/dz_hdp/hosts/dzaratsian-hdp2.field.hortonworks.com', auth=('admin','admin') )
        #req2 = requests.get('http://dzaratsian-hdp1:8080/api/v1/clusters/dz_hdp/hosts/dzaratsian-hdp0.field.hortonworks.com/host_components/DATANODE'.replace('/host_components/DATANODE',''), auth=('admin','admin') )
        req2 = requests.get(host['href'].replace('/host_components/DATANODE',''), auth=(username,password) )
        host_info = json.loads(req2.content)
        
        node_cores.append(host_info['Hosts']['cpu_count'])
        node_ram.append( int(math.floor(host_info['Hosts']['total_mem'] / float(1000*1000))) )


node_count = len(cluster_info['host_components'])
node_cores = min(node_cores)
node_ram   = min(node_ram)

if ((node_cores-1)/5) > 1:
    executor_cores = 5
elif ((node_cores-1)/4) > 1:
    executor_cores = 4
elif ((node_cores-1)/3) > 1:
    executor_cores = 3
else:
    executor_cores = 2


total_cores = node_count * node_cores
total_ram   = node_count * node_ram


yarn_nodemanager_resource_memory_mb  = (node_ram - 2) * 1024
yarn_nodemanager_resource_cpu_vcores = (node_cores - 1)      


executor_cores     = executor_cores  # ~5 or less typically and ideally is a divisor of yarn.nodemanager.resource.cpu-vcores)
executors_per_node = yarn_nodemanager_resource_cpu_vcores / executor_cores 
executor_memory    = ((yarn_nodemanager_resource_memory_mb / executors_per_node))/1024 - 2  # Subtract 2GB for buffer/extra space
num_executors      = (node_count * executors_per_node ) - 1 # Subtract 1 for Driver, since Driver will consume 1 of exector slots
driver_cores       = executor_cores
driver_memory      = executor_memory


# Output Summary
print '\n\n####################################################################\n' + \
    '\nNode Count:         ' + str(node_count) + \
    '\nNode Cores:         ' + str(node_cores) + \
    '\nNode RAM:           ' + str(node_ram) + ' GB' \
    '\n' + \
    '\nTotal Cores:        ' + str(total_cores) + \
    '\nTotal RAM:          ' + str(total_ram) + ' GB' \
    '\n' + \
    '\nexecutors_per_node: ' + str(executors_per_node) + \
    '\n' + \
    '\n--executor-cores:   ' + str(executor_cores) + \
    '\n--executor-memory:  ' + str(executor_memory) + ' GB' \
    '\n--num-executors:    ' + str(num_executors) + \
    '\n\n' + \
    './bin/spark-submit --master yarn --deploy-mode cluster' + ' --driver-cores ' + str(driver_cores) + ' --driver-memory ' + str(driver_memory) + 'G' + ' --executor-memory ' + str(executor_memory) + 'G' + ' --num-executors ' + str(num_executors) + ' --executor-cores ' + str(executor_cores) + \
    '\n\n####################################################################\n' 


# To Do:
# Add unused core count per node
# Add unused memory per node


#ZEND
