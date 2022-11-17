from pyspark.sql import functions as f
import findspark
findspark.init()
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def init_spark(app_name: str):
  spark = SparkSession.builder.appName(app_name).getOrCreate()
  sc = spark.sparkContext
  return spark, sc
  
spark, sc = init_spark('projectB')
df = spark.read.json("data.json")

# prepare data
# calc norm column and select required columns
df_prep = df.withColumn('norm', sum([f.col(j)**2 for j in ['x', 'y', 'z']])**0.5).select('gt', 'norm', 'Arrival_Time', 'User')

transformer_pipeline = Pipeline(stages=[
    StringIndexer(inputCols=["gt"], outputCols=["label"]),
    StringIndexer(inputCols=["User"], outputCols=["user_index"]),
    OneHotEncoder(inputCols=["user_index"], outputCols=["user_vec"]),
    VectorAssembler(inputCols=["norm", "Arrival_Time", "user_vec"], outputCol="features")])

pipe = transformer_pipeline.fit(df_prep)
prepared_df = pipe.transform(df_prep).select('label', 'features')

# ML 
# split data into 70% train, 30% test
train, test = prepared_df.randomSplit([0.7, 0.3])

# create random forest classifier
rfClassifier = RandomForestClassifier(maxDepth=30)
# train model on train data
trainedModel = rfClassifier.fit(train)
print("Trained random forest model on train data successfully")
# run trained model on test data
tested_data = trainedModel.transform(test)
print("Tested model on test data successfully")

# evaluate model, using accuracy metric
print("---")
print(f'Calculating accuracy scores...')
print("---")
correct_pred = tested_data.filter(f.col("prediction") == f.col("label")).count()
total_pred = tested_data.count()
accuray_score = round((correct_pred / total_pred)*100, 2)
print(f'Accuracy score on test set: {accuray_score}%')

trained_data_accuracy = trainedModel.transform(train)
correct_pred = trained_data_accuracy.filter(f.col("prediction") == f.col("label")).count()
total_pred_train = trained_data_accuracy.count()
accuray_score = round((correct_pred / total_pred_train)*100, 2)
print(f'Accuracy score on train set: {accuray_score}%')