
###########################################################################
#
#   Convert Zeppelin Notebook to Python Script
#
#   USAGE: convert_zeppelin.py <zeppelin_notebook_file> <output_python_script>
#
###########################################################################


import json
import codecs
import sys
import re


path_to_zeppelin_nb   = sys.argv[1]
path_to_python_output = open(sys.argv[2], 'w')


notebook = json.load(codecs.open(path_to_zeppelin_nb, 'r', 'utf-8-sig'))


for para in notebook["paragraphs"]:
    
    text = re.sub('^\%[a-zA-Z]+\n', '', para['text'])
    
    if para['config']['editorSetting']['language'] == 'text':
        path_to_python_output.write( "\n'''\n{}\n'''\n".format( text ) )
    
    elif para['config']['editorSetting']['language'] == 'python':
        path_to_python_output.write( "\n{}\n".format( text ) )


path_to_python_output.close()


#ZEND
