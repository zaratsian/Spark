<h3>Text Classification (using Naive Bayes)</h3>
<p>
<strong>Purpose:</strong>
<br>Shows how to use unstructured text data, combined with structured data, to predict an outcome. In this example, I am using labeled data to build a model that predicts airline ratings, based on the customer review of that airline. The rating ranges from 0-10, where 10 is the best rating.
<br>
<br><strong>Input Dataset:</strong> 
<br>Airline Reviews as CSV file in HDFS (which I collected using a custom webcrawler)
<br>
<br><strong>Output Result:</strong> 
<br>A model, which can be used to score new customer reviews. 
<br>I also print out a confusion matrix to show the results (true positives vs misclassifications).
<br>
<br>NOTES:
<br><a href="http://spark.apache.org/docs/latest/api/python/index.html" target="_blank">PySpark Documentation</a>
<br><a href="https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.feature" target="_blank">PySpark Mllib Feature Extraction</a>
<br><a href="https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.classification" target="_blank">PySpark Mllib Classification Algorithms</a>
<br><a href="http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.evaluation" target="_blank">PySpark Mllib Model Evaluation</a>
</p>

note.json is a Zeppelin notebook, which can be imported into your environment and/or viewed from https://www.zeppelinhub.com/viewer/.
