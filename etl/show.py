import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession

def main(in_directory):
    data = spark.read.csv(in_directory, header=True, inferSchema=True)
    data.show()   
    print(data.count())

if __name__ == '__main__':
    in_directory = sys.argv[1]  
    spark = SparkSession.builder.appName('property').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.conf.set("spark.sql.debug.maxToStringFields", 200)
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
