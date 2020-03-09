from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, udf, col
from pyspark.sql.types import ArrayType, DoubleType

from pyspark.ml import Estimator, Model, Transformer, Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoderEstimator, OneHotEncoderModel
from pyspark.ml.linalg import Vectors

from collections import namedtuple

#import flattenVectorUDT as fv
#import FlattenVectorUDT as fv
#import flattenVectorUDT as fv
from flattenVectorUDT import VectorFlattener, VectorReAssembler, VectorFlattenerEstimator

import PandasUDFTransformer.pandas_udf_transformer as pdt
#import FlattenVectorUDT.vector_flattener_transformer as fvt

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
    return VectorFlattener.flattenVectorUDT(df)



if __name__ == '__main__':

    def clean_directory(folder):
        import os
        import shutil

        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


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
              "FlattenVectorUDT",
              timeit.timeit("test_flattenVectorUDT(reps)",
                            setup="from __main__ import get_df, test_flattenVectorUDT; reps = {}; df = get_df(reps)".format(
                                reps),
                            number=7)
              )
        stats = timeit_stats(
              "FlattenVectorUDT",
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
        print("FlattenVectorUDT\t",
              timeit.timeit("test_flattenVectorUDT(reps)",
                           setup="from __main__ import get_df, test_flattenVectorUDT; reps = {}; df = get_df(reps)".format(reps),
                           number=7)
             )
        print("FlattenVectorUDT\t\t\t\t",
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
        for reps in range(100, 1101, 1000):
            stats = get_timeit_stats(f, reps)
            zed[f, reps] = stats
            print("{:20}{:5d}{:6.3f}".format(*stats))

    foo = VectorFlattener().transform(df)
    bar = VectorReAssembler().transform(foo)
    fubarMod = VectorFlattenerEstimator().fit(df)
    df_flattest = fubarMod.transform(df)
    print(fubarMod.outputColumnMap)
    for name, length in zip(fubarMod.getInputCols(), fubarMod.getLengths()):
        print(name, length)
    for name, info in fubarMod.vectorInfo.items():
        print(name, info)
    foo.printSchema()
    bar.printSchema()
    df_flattest.printSchema()
    bar.show(2)
    df_flattest.show(2)
    foo.show(2)
    # zed['fv_500'] = get_timeit_stats("FlattenVectorUDT", reps=500)
    # zed['ex_500'] = get_timeit_stats("extract", reps=500)
    # zed['fv1000'] = get_timeit_stats("FlattenVectorUDT", reps=1000)
    # zed['ex1000'] = get_timeit_stats("extract", reps=1000)
    # for v in zed.values():
    #     print("{:20}{:5d}{:6.3f}".format(*v))

    vfe = VectorFlattenerEstimator()
    pdtransform = pdt.PandasUDFScalarTransformer(function=lambda x: 2.0 * x, returnType="double",
                                                 inputCol="vector[1]", outputCol="vavoom")
    vra = VectorReAssembler()
    pipe0 = Pipeline(stages=[vfe, pdtransform, vra])
    print("Flattener Estimator pipe")
    pipe0.fit(df).transform(df).show(4)

    #pipe = Pipeline(stages=[VectorFlattenerEstimator(), pdtransform, VectorReAssembler()])
    #pipe = Pipeline(stages=[pdtransform, VectorReAssembler()])

    pipeMod = pipe0.fit(df)
    #pipeMod = pipe.fit(foo)

    path = "saved_model"
    clean_directory(path)

    pipeMod.save(path)
    pipeMod2 = PipelineModel.load(path)
    pipeMod2.transform(foo).show()

    pipe1 = Pipeline(stages=[VectorFlattener(), pdtransform, VectorReAssembler()])
    pipeMod1 = pipe1.fit(df)

    path1 = "saved_model_1"
    clean_directory(path1)
    pipeMod1.save(path1)

    print("Flattener Transformer pipe")
    pipeMod3 = PipelineModel.load(path1)
    pipeMod3.transform(df).show