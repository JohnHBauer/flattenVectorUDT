from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, udf, col
from pyspark.sql.types import ArrayType, DoubleType

from pyspark.ml.feature import OneHotEncoderEstimator, OneHotEncoderModel
from pyspark.ml.linalg import Vectors

from collections import namedtuple

import flattenVectorUDT.flattenVectorUDT as fv
import flattenVectorUDT.vector_flattener_transformer as fvt

sc = SparkContext('local[4]', 'FlatTestTime')

spark = SparkSession(sc)
spark.conf.set("spark.sql.execution.arrow.enabled", True)


# copy the two rows in the test dataframe a bunch of times,
# make this small enough for testing, or go for "big data" and be prepared to wait
REPS = 50000

def get_df_0(reps=REPS):
    return sc.parallelize([
        ("assert", Vectors.dense([1, 2, 3]), 1, Vectors.dense([4.1, 5.1])),
        ("require", Vectors.sparse(3, {1: 2}), 2, Vectors.dense([6.2, 7.2])),
    ] * reps).toDF(["word", "vector", "more", "vorpal"])

def get_df(reps=REPS):
    def make_rows(i):
        f = float(i)
        return [
            ("assert", Vectors.sparse(3, [0, 1, 2], [1, 2, 3]), 0, Vectors.dense([f + 0.1, f + 1.1])),
            ("require", Vectors.sparse(3, {1: 2}), 1, Vectors.dense([f + 2.2, f + 3.2])),
        ]
    rows = []
    for i in range(reps):
        rows.extend(make_rows(i))

    df = sc.parallelize(rows).toDF(["word", "vector", "more", "vorpal"])
    ohe = OneHotEncoderEstimator(inputCols=['more'], outputCols=['more__ohe'])
    return ohe.fit(df).transform(df)

def get_bank():
    path = "/Users/john.h.bauer/Downloads/bank-additional/bank-additional-full.csv"
    df = spark.read.csv(path, header=True, inferSchema=True, sep=";")
    df.printSchema()
    return df
#df = get_df(REPS)

def extract(row):
    return (row.word, ) + tuple(row.vector.toArray().tolist(),) + (row.more,) + tuple(row.vorpal.toArray().tolist(),)

def test_extract(df, reps):
    return df.rdd.map(extract).toDF(['word', 'vector__0', 'vector__1', 'vector__2', 'more', 'vorpal__0', 'vorpal__1'])

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)

def test_to_array(df, reps):
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

def test_flatten(df, reps):
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
def test_ith(df, reps):
    return df.select(select)

def test_flattenVectorUDT(df, reps):
    return fv.flattenVectorUDT(df)

if __name__ == '__main__':
    import timeit

    df = get_df(10)
    df.printSchema()
    df.show(4)
   # df = get_bank()
    # make sure these work as intended
    #test_ith(df, 10).show(4)
    #test_flatten(df, 10).show(4)
    #test_to_array(df, 10).show(4)
    #test_extract(df, 10).show(4)
    test_flattenVectorUDT(df, 10).show(4)

    timeit_stats = namedtuple("timeit_stats", ["function", "repetitions", "time"])


    def get_timeit_stats(function, reps=REPS):
        time = timeit.timeit("test_{}(df, reps)".format(function),
                             setup="from __main__ import get_df, test_{}; reps = {}; df = get_df(reps)".format(function, reps),
                             number=7)
        return timeit_stats(function, reps, time)


    def get_timeit_stats_all(reps=REPS):
        stats = timeit_stats(
              "i_th",
              reps,
              timeit.timeit("test_ith(reps)",
                            setup="from __main__ import get_df, test_ith; reps = {}; df = get_df(reps)".format(reps),
                            number=7)
              )
        stats = timeit_stats(
              "flattenVectorUDT",
              timeit.timeit("test_flattenVectorUDT(reps)",
                            setup="from __main__ import get_df, test_flattenVectorUDT; reps = {}; df = get_df(reps)".format(
                                reps),
                            number=7)
              )
        stats = timeit_stats(
              "flattenVectorUDT",
              timeit.timeit("test_flatten(reps)",
                            setup="from __main__ import get_df, test_flatten; reps = {}; df = get_df(reps)".format(
                                reps),
                            number=7)
              )
        stats = timeit_stats(
              "to_array",
              timeit.timeit("test_to_array(reps)",
                            setup="from __main__ import get_df, test_to_array; reps = {}; df = get_df(reps)".format(
                                reps),
                            number=7)
              )
        stats = timeit_stats(
              "extract",
              timeit.timeit("test_extract(reps)",
                            setup="from __main__ import get_df, test_extract; reps = {}; df = get_df(reps)".format(
                                reps),
                            number=7)
              )

    def time_all(reps=REPS):
        print("-" * 50, "repetitions =", reps)
        print("i_th\t\t\t\t",
              timeit.timeit("test_ith(reps)",
                           setup="from __main__ import get_df, test_ith; reps = {}; df = get_df(reps)".format(reps),
                           number=7)
             )
        print("flattenVectorUDT\t",
              timeit.timeit("test_flattenVectorUDT(reps)",
                           setup="from __main__ import get_df, test_flattenVectorUDT; reps = {}; df = get_df(reps)".format(reps),
                           number=7)
             )
        print("flattenVectorUDT\t\t\t\t",
              timeit.timeit("test_flatten(reps)",
                           setup="from __main__ import get_df, test_flatten; reps = {}; df = get_df(reps)".format(reps),
                           number=7)
             )
        print("to_array\t\t\t",
              timeit.timeit("test_to_array(reps)",
                           setup="from __main__ import get_df, test_to_array; reps = {}; df = get_df(reps)".format(reps),
                           number=7)
             )
        print("extract\t\t\t\t",
              timeit.timeit("test_extract(reps)",
                           setup="from __main__ import get_df, test_extract; reps = {}; df = get_df(reps)".format(reps),
                           number=7)
             )

    # for reps in range(1000, 3001, 1000):
    #     time_all(reps)

    zed = {}
    for f in ("flattenVectorUDT", "extract"):
        for reps in range(100, 1101, 100):
            stats = get_timeit_stats(f, reps)
            zed[f, reps] = stats
            print("{:20}{:5d}{:6.3f}".format(*stats))

    foo = fvt.VectorFlattener().transform(df)
    bar = fvt.VectorReAssembler().transform(foo)
    foo.printSchema()
    bar.printSchema()
    bar.show(2)
    # zed['fv_500'] = get_timeit_stats("flattenVectorUDT", reps=500)
    # zed['ex_500'] = get_timeit_stats("extract", reps=500)
    # zed['fv1000'] = get_timeit_stats("flattenVectorUDT", reps=1000)
    # zed['ex1000'] = get_timeit_stats("extract", reps=1000)
    # for v in zed.values():
    #     print("{:20}{:5d}{:6.3f}".format(*v))

