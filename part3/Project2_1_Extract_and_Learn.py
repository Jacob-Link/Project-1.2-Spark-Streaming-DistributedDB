from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import os

# schema for the streamed data
SCHEMA = StructType([StructField("Arrival_Time",LongType(),True),
                     StructField("Creation_Time",LongType(),True),
                     StructField("Device",StringType(),True),
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

# schema for the all the data prepared for use by the model
saved_data_schema = StructType([StructField("label",FloatType(),True),
                     StructField("features",VectorUDT(),True)])

spark = SparkSession.builder.appName('demo_app')\
    .config("spark.kryoserializer.buffer.max", "512m")\
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "activities"

# create global model and global df which will be store prepared data for the model and the most updated model
emp_RDD = spark.sparkContext.emptyRDD()
unbounded_df =  spark.createDataFrame(data = emp_RDD, schema = saved_data_schema) # empty df to start off
correct = 0  # global var counting the correct predictions made
total = 0 # global var counting the total predictions made
model = 0
flag = 1

streaming = spark.readStream\
                  .format("kafka")\
                  .option("kafka.bootstrap.servers", kafka_server)\
                  .option("subscribe", topic)\
                  .option("startingOffsets", "earliest")\
                  .option("failOnDataLoss",False)\
                  .option("maxOffsetsPerTrigger", 100000)\
                  .load()\
                  .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")
  
def foreach_batch_function(df, epoch_id):
  # access all global variables
  global unbounded_df
  global correct
  global total
  global model
  global flag

  # string indexer alternative for spark 2.4 done manually for gt  (created dict)
  gt_ = ['null', 'stairsup', 'walk', 'stand', 'sit', 'bike', 'stairsdown']
  index_gt = [str(num) for num in range(len(gt_))]
  gtdict = dict(zip(gt_, index_gt))

  # create norm column and update gt to be float of label and not string
  df = df.withColumn("label", f.col("gt")).replace(to_replace=gtdict, subset=["label"])\
    .withColumn("label", f.col("label").cast(FloatType()))\
    .withColumn('norm', sum([f.col(j)**2 for j in ['x', 'y', 'z']])**0.5)\
                .select('gt', 'norm', 'Arrival_Time', 'User', 'label')

  
  # string indexer alternative for spark 2.4 done manually done for user
  users_ = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
  user_columns = [f.when(f.col("User") == uid, 1).otherwise(0).alias("User_ID_" + uid) for uid in users_]
  user_fields = ['User_ID_' + str(i) for i in users_]

  df = df.select(*user_columns, 'gt', 'norm', 'Arrival_Time', 'User', 'label')

  vector_columns = ["norm", "Arrival_Time"] + user_fields
  assembler = VectorAssembler(inputCols=vector_columns, outputCol='features')
  df = assembler.transform(df).select('label', 'features')  # batch data ready for prediction

  test_data = df  # the test data is always the batch data the stream has recieved
  
  if total == 0: # first prediction, predict all batch data as first label
    test_predicted = test_data.withColumn('pred', f.lit(3.0))  # predict all data to be 3.0

    correct_batch = test_predicted.filter(f.col("pred") == f.col("label")).count()
    rows_in_batch = test_predicted.count()
    
    correct += correct_batch
    total += rows_in_batch
  
  else: # not first batch, use global model to predict
    test_predicted = model.transform(test_data)
    
    correct_batch = test_predicted.filter(f.col("prediction") == f.col("label")).count()
    rows_in_batch = test_predicted.count()
    
    correct += correct_batch
    total += rows_in_batch
    
  print("(Batch:",epoch_id + 1, ") Current Accuracy: ",round(correct/total*100, 2),"% --- (total predicted: ", total, ", total correct: ",correct, ", correct in batch: ", correct_batch, ")")
  
  if total % 100000 != 0:  # end running of the code, no more data in the source
      print()
      print("----- No more data being streamed, FINAL ACCURACY: ",round(correct/total*100, 2), "% -----")
      print()
      flag = 0 # will cause an exit from the while loop in main code and stop the stream

  elif (epoch_id + 1) < 40:  # Update model after predicting each batch until trained on 4,000,000 rows. Once achieved, use model for future predictions.
    # create random forest classifier
    rfClassifier = RandomForestClassifier(numTrees=10, maxDepth=30, maxBins =70)
    
    # train model on the new unbounded df (after used old model to predict the current batch)
    unbounded_df = unbounded_df.union(df)  # update unbounded_df holding all data, works spark 2.4
    
    # train model for next batch
    model = rfClassifier.fit(unbounded_df)

stream_obj = streaming.writeStream.trigger(processingTime='3 seconds').foreachBatch(foreach_batch_function).start()

while(flag): # used instead of awaitTermination function which wasnt working well in the forEachBatch sink
    pass

stream_obj.stop()

