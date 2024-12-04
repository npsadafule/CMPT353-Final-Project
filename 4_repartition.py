"Script to  repartion the data and write it to a new directory while reducing its size from 1.28GB to 237MB"

import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types

def main(in_directory, out_directory):
    data = spark.read.csv(in_directory)
    data = data.repartition(1000)
    data.write.csv(out_directory, mode='overwrite', compression="gzip")


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('property').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
