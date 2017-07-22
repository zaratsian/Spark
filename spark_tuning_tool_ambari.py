
###################################################################################################
#
#   Spark Tuning Tool (via Ambari)
#   Dan Zaratsian
#
#   This tool will extract cluster information from your cluster (via Ambari APIs), then the 
#   code will use these extracted parameters in order to calculate the optimal Spark configuration.
#
#   Usage: spark_tuning_tool_ambari.py --ambari_hostname=dzaratsian_hdp.field.hortonworks.com --ambari_port --ambari_cluster_name --username=admin --password=admin
#
# https://github.com/apache/ambari/blob/trunk/ambari-server/docs/api/v1/index.md#resources
###################################################################################################


import sys,re
import getopt
import requests
import json
import math


try:
    opts, args = getopt.getopt(sys.argv[1:], 'x', ['ambari_hostname=', 'ambari_port=', 'cluster_name=', 'username=', 'password='])
    
    ambari_hostname     = [opt[1] for opt in opts if opt[0]=='--ambari_hostname'][0]
    ambari_port         = [opt[1] for opt in opts if opt[0]=='--ambari_port'][0]
    cluster_name        = [opt[1] for opt in opts if opt[0]=='--ambari_cluster_name'][0]
    username            = [opt[1] for opt in opts if opt[0]=='--username'][0]
    password            = [opt[1] for opt in opts if opt[0]=='--password'][0]
except:
    print '\n\n[ USAGE ] spark_tuning_tool_ambari.py --ambari_hostname=<hostname> --ambari_port=<port> --cluster_name=<cluster_name> --username=<string> --password=<string>\n\n'
    sys.exit(1)



def get_datanode_parameters(ambari_hostname, ambari_port, cluster_name, username, password):
    
    nodes_names = []
    node_cores  = []
    node_ram    = []  
    
    url = 'http://' + str(ambari_hostname) + ':' + str(ambari_port) + '/api/v1/clusters/' + str(cluster_name) + '/services/HDFS/components/DATANODE'
    print '[ INFO ] Collection data from ' + str(url)
    req = requests.get(url, auth=(username,password) )
    
    if req.status_code == 200:
        cluster_info = json.loads(req.content)
        
        for host in cluster_info['host_components']:
            url = host['href'].replace('/host_components/DATANODE','')
            print '[ INFO ] Collection data from ' + str(url)
            req2 = requests.get(url, auth=(username,password) )
            host_info = json.loads(req2.content)
            
            nodes_names.append(host['HostRoles']['host_name'])
            node_cores.append(host_info['Hosts']['cpu_count'])
            node_ram.append( int(math.floor(host_info['Hosts']['total_mem'] / float(1000*1000))) )
    
    node_count = len(nodes_names)
    node_cores = min(node_cores)
    node_ram   = min(node_ram)
    
    return (node_count, node_cores, node_ram)


node_count, node_cores, node_ram = get_datanode_parameters(ambari_hostname, ambari_port, cluster_name, username, password)


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


def get_yarn_parameters(ambari_hostname, ambari_port, cluster_name, username, password):
    
    url = 'http://' + str(ambari_hostname) + ':' + str(ambari_port) + '/api/v1/clusters/' + str(cluster_name) + '/configurations/service_config_versions?service_name=YARN&service_config_version=1'
    print '[ INFO ] Collection data from ' + str(url)
    req = requests.get(url, auth=(username,password))
    
    if req.status_code == 200:
        cluster_info = json.loads(req.content)
    
    yarn_site = [config for config in cluster_info['items'][0]['configurations'] if config['type']=='yarn-site']
    
    yarn_nodemanager_resource_memory_mb   = int(yarn_site[0]['properties']['yarn.nodemanager.resource.memory-mb'])
    #yarn_nodemanager_resource_memory_mb  = (node_ram - 2) * 1024
    
    yarn_nodemanager_resource_cpu_vcores  = int(yarn_site[0]['properties']['yarn.nodemanager.resource.cpu-vcores'])
    #yarn_nodemanager_resource_cpu_vcores = (node_cores - 1)
    
    return (yarn_nodemanager_resource_memory_mb, yarn_nodemanager_resource_cpu_vcores)


yarn_nodemanager_resource_memory_mb, yarn_nodemanager_resource_cpu_vcores = get_yarn_parameters(ambari_hostname, ambari_port, cluster_name, username, password)


executor_cores     = executor_cores  # ~5 or less typically and ideally is a divisor of yarn.nodemanager.resource.cpu-vcores)
executors_per_node = yarn_nodemanager_resource_cpu_vcores / executor_cores 
executor_memory    = ((yarn_nodemanager_resource_memory_mb / executors_per_node))/1024 - 2  # Subtract 2GB for buffer/extra space
num_executors      = (node_count * executors_per_node ) - 1 # Subtract 1 for Driver, since Driver will consume 1 of exector slots
driver_cores       = executor_cores
driver_memory      = executor_memory


# Output Summary
print '\n\n####################################################################\n' + \
    '\nNode Count:                      ' + str(node_count) + \
    '\nNode Cores:                      ' + str(node_cores) + \
    '\nNode RAM:                        ' + str(node_ram) + ' GB' \
    '\n' + \
    '\nTotal Cores:                     ' + str(total_cores) + \
    '\nTotal RAM:                       ' + str(total_ram) + ' GB' \
    '\n' + \
    '\nYARN Nodemanager RAM (mb):       ' + str(yarn_nodemanager_resource_memory_mb) + \
    '\nYARN Nodemanager CPU Vcores:     ' + str(yarn_nodemanager_resource_cpu_vcores) + \
    '\n' + \
    '\nexecutors_per_node:              ' + str(executors_per_node) + \
    '\n' + \
    '\n--executor-cores:                ' + str(executor_cores) + \
    '\n--executor-memory:               ' + str(executor_memory) + ' GB' \
    '\n--num-executors:                 ' + str(num_executors) + \
    '\n\n' + \
    './bin/spark-submit --master yarn --deploy-mode cluster' + ' --driver-cores ' + str(driver_cores) + ' --driver-memory ' + str(driver_memory) + 'G' + ' --executor-memory ' + str(executor_memory) + 'G' + ' --num-executors ' + str(num_executors) + ' --executor-cores ' + str(executor_cores) + \
    '\n\n' + \
    '/usr/hdp/current/spark2-client/bin/pyspark --master yarn --deploy-mode client' + ' --driver-cores ' + str(driver_cores) + ' --driver-memory ' + str(driver_memory) + 'G' + ' --executor-memory ' + str(executor_memory) + 'G' + ' --num-executors ' + str(num_executors) + ' --executor-cores ' + str(executor_cores) + \
    '\n\n####################################################################\n' 



# Add unused core count per node
# Add unused memory per node


#ZEND
