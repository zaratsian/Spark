

################################################################################################
# http://spark.apache.org/docs/2.0.2/ml-guide.html
# http://spark.apache.org/docs/2.0.2/sql-programming-guide.html
# http://spark.apache.org/docs/2.0.2/ml-features.html#tf-idf
# http://spark.apache.org/docs/2.0.2/api/python/pyspark.ml.html#pyspark.ml.feature.IDF
################################################################################################


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
import re

################################################################################################
#
#   Import Rawdata
#
################################################################################################

rawdata = spark.read.load("/airlines.csv", format="csv", header=True)
rawdata = rawdata.fillna({'review': ''})                               # Replace nulls with blank string
rawdata = rawdata.withColumn("uid", monotonically_increasing_id())     # Create Unique ID

# Show rawdata (as DataFrame)
rawdata.show(10)

# Print data types
for type in rawdata.dtypes:
    print type

target = rawdata.select(rawdata['rating'].cast(IntegerType()))
target.dtypes

################################################################################################
#
#   Text Pre-processing (consider using one or all of the following):
#       - Remove common words (with stoplist)
#       - Handle punctuation
#       - lowcase/upcase
#       - Stemming
#       - Part-of-Speech Tagging (nouns, verbs, adj, etc.)
#
################################################################################################

def cleanup_text(record):
    text  = record[8]
    uid   = record[9]
    words = text.split()
    
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = ['']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out

udf_cleantext = udf(cleanup_text , ArrayType(StringType()))
clean_text = rawdata.withColumn("words", udf_cleantext(struct([rawdata[x] for x in rawdata.columns])))

#tokenizer = Tokenizer(inputCol="description", outputCol="words")
#wordsData = tokenizer.transform(text)

################################################################################################
#
#   Generate TFIDF
#
################################################################################################

# Term Frequency Vectorization  - Option 1 (Using hashingTF): 
#hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
#featurizedData = hashingTF.transform(clean_text)

# Term Frequency Vectorization  - Option 2 (CountVectorizer)    : 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 1000)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

################################################################################################
#
#   LDA Clustering - Find Data-driven Topics
#
################################################################################################

lda = LDA(k=25, seed=123, optimizer="em", featuresCol="features")

ldamodel = lda.fit(rescaledData)

#model.isDistributed()
#model.vocabSize()

ldatopics = ldamodel.describeTopics()
ldatopics.show(25)

def map_termID_to_Word(termIndices):
    words = []
    for termID in termIndices:
        words.append(vocab_broadcast.value[termID])
    
    return words

udf_map_termID_to_Word = udf(map_termID_to_Word , ArrayType(StringType()))
ldatopics_mapped = ldatopics.withColumn("topic_desc", udf_map_termID_to_Word(ldatopics.termIndices))
ldatopics_mapped.select(ldatopics_mapped.topic, ldatopics_mapped.topic_desc).show(25,False)

ldaResults = ldamodel.transform(rescaledData)

ldaResults.show()

################################################################################################
#
#   Breakout LDA Topics for Modeling and Reporting
#
################################################################################################

def breakout_array(index_number, record):
    vectorlist = record.tolist()
    return vectorlist[index_number]

udf_breakout_array = udf(breakout_array, FloatType())
enrichedData = ldaResults                                                                   \
        .withColumn("Topic_12", udf_breakout_array(lit(12), ldaResults.topicDistribution))  \
        .withColumn("topic_20", udf_breakout_array(lit(2), ldaResults.topicDistribution))            

enrichedData.show()

#enrichedData.agg(max("Topic_12")).show()

################################################################################################
#
#   Register Table for SparkSQL
#
################################################################################################

enrichedData.createOrReplaceTempView("enrichedData")

spark.sql("SELECT id, airline, date, rating, topic_12 FROM enrichedData")

spark.sql("SELECT id, airline, date, rating, topic_20 FROM enrichedData")


#ZEND
