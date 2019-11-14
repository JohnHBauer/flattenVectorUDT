from pyspark.context import SparkContext
from pyspark.sql import Row, SQLContext, SparkSession
from pyspark.sql.functions import lit, udf, col
from pyspark.sql.types import ArrayType, DoubleType

import flattenVectorUDT as fv

import pyspark.sql.dataframe
from pyspark.sql.functions import pandas_udf, PandasUDFType

sc = SparkContext('local[4]', 'FlatTestTime')

spark = SparkSession(sc)
spark.conf.set("spark.sql.execution.arrow.enabled", True)

from pyspark.ml.linalg import Vectors

# copy the two rows in the test dataframe a bunch of times,
# make this small enough for testing, or go for "big data" and be prepared to wait
REPS = 50000

def get_df(reps=REPS):
    return sc.parallelize([
        ("assert", Vectors.dense([1, 2, 3]), 1, Vectors.dense([4.1, 5.1])),
        ("require", Vectors.sparse(3, {1: 2}), 2, Vectors.dense([6.2, 7.2])),
    ] * reps).toDF(["word", "vector", "more", "vorpal"])

df = get_df(REPS)

def extract(row):
    return (row.word, ) + tuple(row.vector.toArray().tolist(),) + (row.more,) + tuple(row.vorpal.toArray().tolist(),)

def test_extract():
    return df.rdd.map(extract).toDF(['word', 'vector__0', 'vector__1', 'vector__2', 'more', 'vorpal__0', 'vorpal__1'])

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)

def test_to_array():
    df_to_array = df.withColumn("xs", to_array(col("vector"))) \
        .select(["word"] + [col("xs")[i] for i in range(3)] + ["more", "vorpal"]) \
        .withColumn("xx", to_array(col("vorpal"))) \
        .select(["word"] + ["xs[{}]".format(i) for i in range(3)] + ["more"] + [col("xx")[i] for i in range(2)])
    return df_to_array

# pack up to_array into a tidy function
def flatten(df, vector, vlen):
    fieldNames = df.schema.fieldNames()
    if vector in fieldNames:
        names = []
        for fieldname in fieldNames:
            if fieldname == vector:
                names.extend([col(vector)[i] for i in range(vlen)])
            else:
                names.append(col(fieldname))
        return df.withColumn(vector, to_array(col(vector)))\
                 .select(names)
    else:
        return df

def test_flatten():
    dflat = flatten(df, "vector", 3)
    dflat2 = flatten(dflat, "vorpal", 2)
    return dflat2

def ith_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

ith = udf(ith_, DoubleType())

select = ["word"]
select.extend([ith("vector", lit(i)) for i in range(3)])
select.append("more")
select.extend([ith("vorpal", lit(i)) for i in range(2)])

# %% timeit ...
def test_ith():
    return df.select(select)

def test_flattenVectorUDT():
    return fv.flattenVectorUDT(df)

if __name__ == '__main__':
    import timeit

    # make sure these work as intended
    test_ith().show(4)
    test_flatten().show(4)
    test_to_array().show(4)
    test_extract().show(4)
    test_flattenVectorUDT().show(4)

    print("i_th\t\t",
          timeit.timeit("test_ith()",
                       setup="from __main__ import test_ith",
                       number=7)
         )
    print("flattenVectorUDT\t\t",
          timeit.timeit("test_flattenVectorUDT()",
                       setup="from __main__ import test_flattenVectorUDT",
                       number=7)
         )
    print("flatten\t\t",
          timeit.timeit("test_flatten()",
                       setup="from __main__ import test_flatten",
                       number=7)
         )
    print("to_array\t",
          timeit.timeit("test_to_array()",
                       setup="from __main__ import test_to_array",
                       number=7)
         )
    print("extract\t\t",
          timeit.timeit("test_extract()",
                       setup="from __main__ import test_extract",
                       number=7)
         )
