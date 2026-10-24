from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("cubo_izquierdo")
    .master("local[*]")
    .config("spark.local.dir", "/home/jupyter-lquijano/spark_tmp")
    .config("spark.driver.memory", "800g")
    .config("spark.driver.maxResultSize", "64g")
    .config("spark.sql.shuffle.partitions", "960")
    .config("spark.default.parallelism", "288")
    .config("spark.sql.maxPlanStringLength", "8192")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "250g")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max", "1g")
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:G1HeapRegionSize=32m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
