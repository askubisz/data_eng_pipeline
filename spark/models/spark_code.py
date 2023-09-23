# Import libraries
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression, RandomForestRegressor, GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F


# Create a SparkSession
spark = SparkSession.builder.appName("DecisionTreeRegression").getOrCreate()

########## LOAD, PREPARE AND SPLIT THE DATA

# Load data from MySQL database
jdbc_url = "jdbc:mysql://scc413-mysqldb:3306/us_census"
connection_properties = {
    "user": "root",
    "password": "example",
    "driver": "com.mysql.jdbc.Driver"
}
data = spark.read.jdbc(url=jdbc_url, table="earning_data", properties=connection_properties)

# Prepare data for machine learning

# Identify categorical columns
categorical_cols = ["sex", "schl", "mar"]

# Encode categorical columns using StringIndexer
indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed", handleInvalid="keep")
            for col in categorical_cols]
# Prepare assembler to get features into one column
assembler = VectorAssembler(inputCols=[col+"_indexed" for col in categorical_cols] + ["agep"],
                            outputCol="features")

# Index the columns with string indexer
string_indexer_model1 = indexers[-1].fit(data)
data = string_indexer_model1.transform(data)
string_indexer_model2 = indexers[-2].fit(data)
data = string_indexer_model2.transform(data)
string_indexer_model3 = indexers[-3].fit(data)
data = string_indexer_model3.transform(data)

# Split data into training and test sets
(training_data, test_data) = data.randomSplit([0.7, 0.3], seed=1234)

######## CHECK REGRESSION MODELS

# Train possible regression models on the training data
dt = DecisionTreeRegressor(featuresCol="features", labelCol="wagp")
model1 = dt.fit(assembler.transform(training_data).select("features", "wagp"))

lr = LinearRegression(featuresCol="features", labelCol="wagp")
model2 = lr.fit(assembler.transform(training_data).select("features", "wagp"))

rf = RandomForestRegressor(featuresCol="features", labelCol="wagp")
model3 = rf.fit(assembler.transform(training_data).select("features", "wagp"))

glr = GeneralizedLinearRegression(featuresCol="features", labelCol="wagp", family="gamma")
model4 = glr.fit(assembler.transform(training_data).select("features", "wagp"))

# Evaluate the models
DTtest_predictions = model1.transform(assembler.transform(test_data).select("features", "wagp"))
LRtest_predictions = model2.transform(assembler.transform(test_data).select("features", "wagp"))
RFtest_predictions = model3.transform(assembler.transform(test_data).select("features", "wagp"))
GLRtest_predictions = model4.transform(assembler.transform(test_data).select("features", "wagp"))

mae_evaluator = RegressionEvaluator(labelCol="wagp", predictionCol="prediction", metricName="mae")
rmse_evaluator = RegressionEvaluator(labelCol="wagp", predictionCol="prediction", metricName="rmse")

DTmae = mae_evaluator.evaluate(DTtest_predictions)
DTrmse = rmse_evaluator.evaluate(DTtest_predictions)
LRmae = mae_evaluator.evaluate(LRtest_predictions)
LRrmse = rmse_evaluator.evaluate(LRtest_predictions)
RFmae = mae_evaluator.evaluate(RFtest_predictions)
RFrmse = rmse_evaluator.evaluate(RFtest_predictions)
GLRmae = mae_evaluator.evaluate(GLRtest_predictions)
GLRrmse = rmse_evaluator.evaluate(GLRtest_predictions)

print("Decision Tree Mean absolute percentage error on test data = %g" % DTmae) # 30937.3
print("Decision Tree Root mean squared error on test data = %g" % DTrmse) # 56243.6
print("Linear Regression Mean absolute percentage error on test data = %g" % LRmae) # 35219.5
print("Linear Regression Root mean squared error on test data = %g" % LRrmse) # 60874.1
print("Random Forest Mean absolute percentage error on test data = %g" % RFmae) # 30983.2
print("Random Forest Root mean squared error on test data = %g" % RFrmse) # 56383
print("GLR Mean absolute percentage error on test data = %g" % GLRmae) # 35462.2
print("GLR Root mean squared error on test data = %g" % GLRrmse) # 60964.6

# Save the best model
model3.write().overwrite().save('models/model')

########## MAKE PREDICTIONS ON NEW DATA

# Create variables that will store possible values for predictions into MySQL table
sexValues=[1,2]
agepValues = range(0,100)
schlValues = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
marValues = range(1,6)

col1 = spark.createDataFrame([(value,) for value in sexValues], ["sex"])
col2 = spark.createDataFrame([(value,) for value in agepValues], ["agep"])
col3 = spark.createDataFrame([(value,) for value in schlValues], ["schl"])
col4 = spark.createDataFrame([(value,) for value in marValues], ["mar"])

# Create a DataFrame with the Cartesian product of the unique values
input_data = col1.crossJoin(col2).crossJoin(col3).crossJoin(col4)
input_data = input_data.withColumn('index', F.monotonically_increasing_id())

# Create copy of DataFrame for the future (joining table with the predictions)
input_data.createOrReplaceTempView("my_table")
new_df = spark.sql("SELECT * FROM my_table")

# Index the columns
input_data = string_indexer_model1.transform(input_data)
input_data = string_indexer_model2.transform(input_data)
input_data = string_indexer_model3.transform(input_data)

# Make predictions
predictions = model3.transform(assembler.transform(input_data).select("features", "index"))

# Extract predictions and index
data = predictions.select('prediction', 'index')

# Join tables to get unindexed data with their respective prediction
result = new_df.join(data, new_df.index == data.index, 'left') \
               .select('sex', 'agep', 'schl', 'mar', 'prediction')

######### SAVE INTO MYSQL DATABASE

# Prepare a properties object with the JDBC connection details
jdbc_url = "jdbc:mysql://scc413-mysqldb:3306/us_census"
connection_properties = {
    "user": "root",
    "password": "example",
    "driver": "com.mysql.jdbc.Driver"
}

# Write the predictions to the MySQL table
result.write \
    .option("truncate", "true") \
    .jdbc(jdbc_url, "predictions", mode="append", properties=connection_properties)

# Stop the SparkSession
spark.stop()
