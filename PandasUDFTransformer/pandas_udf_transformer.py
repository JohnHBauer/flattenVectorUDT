import codecs
import pickle

from abc import ABC, abstractmethod

from pyspark import keyword_only

from pyspark.ml import Estimator, Model, Transformer, Pipeline
from pyspark.rdd import PythonEvalType
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import pandas_udf, lit, col

from flattenVectorUDT.flattenVectorUDT import VectorDataType

# TODO: HasFunction, HasReturnType, HasFunctionType -- also consider non-trivial default values

class HasFunction(Params):
    """
    Mixin for param nu: proportion of vectors to use as support
    """

    function = Param(Params._dummy(), "function", "function object, or its encoded string representation", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasFunction, self).__init__()

    def setFunction(self, value):
        """
        Sets the value of :py:attr:`function`.  If value is callable, it will be encoded as a string to allow
        PandasUDFTransformer objects to be saved and loaded.
        """
        return self._set(function=self.encode_function(value))

    def getFunction(self):
        """
        Gets the value of function or its default value.  Note that this returns an encoded string, which must be
        decoded with `PandasUDFTransformer.decode_function` before use.
        """
        return self.getOrDefault(self.function)

    @staticmethod
    def encode_function(function):
        """Encode a callable as a string which can be used as an argument to PandasUDFTransformer.

        The function is pickled to `bytes`, encoded with base64 to prevent invalid characters, then decoded to a string.

        `<https://stackoverflow.com/questions/30469575/how-to-pickle-and-unpickle-to-portable-string-in-python-3>`_
        If a string is supplied, it is presumed to have been previously encoded and will be returned unchanged.
        For example, after being saved, load will create a new object using the encoded function.
        """
        if callable(function):
            # make a string suitable for read/write so Transformer can be saved and loaded
            return codecs.encode(pickle.dumps(function), "base64").decode()
        elif isinstance(function, basestring):
            # function had better be a string which is already an encoded function, or caller beware
            return function
        else:
            raise ValueError("function argument must be callable, or a string which has already been encoded.")

    @staticmethod
    def decode_function(function):
        """Decode and un-pickle a string, presumed to have been encoded with `PandasUDFTransformer.encode_function`.

        The string is encoded to `bytes`, base64-decoded, then un-pickled.
        """
        try:
            f = pickle.loads(codecs.decode(function.encode(), "base64"))
        except:
            raise ValueError("encoded function required.")
        if not callable(f):
            raise ValueError("decoded string must be callable, i.e. a function")
        # could check for 1 or 2 positional arguments ???
        return f

class HasReturnType(Params):
    """
    Mixin for param returnType: DDL-formatted type string describing the schema to use on the returned dataframe.
    """

    returnType = Param(Params._dummy(), "returnType", "DDL-formatted type string describing the schema of the returned dataframe",
                typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasReturnType, self).__init__()

    def setReturnType(self, value):
        """
        Sets the value of :py:attr:`returnType`.
        """
        return self._set(returnType=value)

    def getReturnType(self):
        """
        Gets the value of returnType or its default value.
        """
        return self.getOrDefault(self.returnType)


class HasFunctionType(Params):
    """
    Mixin for param functionType: constant value, interpreted as defined by PythonEvalType.
    """

    functionType = Param(Params._dummy(), "functionType", "constant value, interpreted as defined by PythonEvalType",
                         typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasFunctionType, self).__init__()

    def setFunctionType(self, value):
        """
        Sets the value of :py:attr:`functionType`.

        Valid values are defined in PythonEvalType:

        SQL_SCALAR_PANDAS_UDF = 200
        SQL_GROUPED_MAP_PANDAS_UDF = 201
        SQL_GROUPED_AGG_PANDAS_UDF = 202
        SQL_WINDOW_AGG_PANDAS_UDF = 203
        """
        if value not in (PythonEvalType.SQL_SCALAR_PANDAS_UDF,
                         PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF,
                         PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
                         PythonEvalType.SQL_WINDOW_AGG_PANDAS_UDF,
                         ):
            raise ValueError("Function type must be some kind of Pandas_UDF")

        return self._set(functionType=value)

    def getFunctionType(self):
        """
        Gets the value of functionType or its default value.
        """
        return self.getOrDefault(self.functionType)


class HasGroupBy(Params):
    """
    Mixin for param returnType: DDL-formatted type string describing the schema to use on the returned dataframe.
    """

    groupBy = Param(Params._dummy(), "groupBy", "List of columns to group by for GROUPED_MAP or GROUPED_AGG",
                typeConverter=TypeConverters.toListString)

    def __init__(self):
        super(HasGroupBy, self).__init__()

    def setGroupBy(self, value):
        """
        Sets the value of :py:attr:`returnType`.
        """
        return self._set(groupBy=value)

    def getGroupBy(self):
        """
        Gets the value of returnType or its default value.
        """
        return self.getOrDefault(self.groupBy)


class PandasUDFTransformer(ABC, Transformer, HasFunction, HasFunctionType, HasReturnType,
                           DefaultParamsReadable, DefaultParamsWritable):
    """Allows PySpark User-Defined Function to be used in a Pipeline.

    Parameters
    ----------
    function:
        must be a callable object which can be pickled, or a string representation thereof obtained with
        HasFunction.encode_function.
    functionType:
        must be a value defined by the enumeration PythonEvalType.
    returnType:
        DDL-formatted string describing the schema of the returned Spark dataframe.
        For example "id long, v double"
    """

    @keyword_only
    @abstractmethod
    def __init__(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF):
        pass

    @keyword_only
    @abstractmethod
    def setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF):
        """
        setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF)
        """
        raise NotImplementedError()

    @abstractmethod
    def _transform(self, data):
        raise NotImplementedError()

    @property
    def pandas_udf(self):
        function = self.getFunction()
        returnType = self.getReturnType()
        functionType = self.getFunctionType()

        # f = pickle.loads(codecs.decode(function.encode(), "base64"))
        f = self.decode_function(function)

        if not callable(f):
            raise ValueError("Decoded function parameter is not callable.")

        return pandas_udf(f=f,
                          returnType=returnType,
                          functionType=functionType,
                          )


class PandasUDFGroupedTransformer(PandasUDFTransformer, HasGroupBy,
                           DefaultParamsReadable, DefaultParamsWritable):
    """Allows PySpark User-Defined Function to be used in a Pipeline.

    Parameters
    ----------
    function:
        must be a callable object which can be pickled, or a string representation thereof obtained with
        HasFunction.encode_function.
    functionType:
        must be a value defined by the enumeration PythonEvalType.
    returnType:
        DDL-formatted string describing the schema of the returned Spark dataframe.
        For example "id long, v double"
    """
    @keyword_only
    def __init__(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[]):
        super(Transformer, self).__init__()

        self._setDefault(function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[])

        kwargs = self._input_kwargs

        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[]):
        """
        setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[])
        """
        kwargs = self._input_kwargs
        kwargs['function'] = self.encode_function(function)
        self._set(**kwargs)
        return self

    def _transform(self, data):
        function = self.getFunction()
        returnType = self.getReturnType()
        functionType = self.getFunctionType()
        groupBy = self.getGroupBy()

        f = self.decode_function(function)
        
        if not callable(f):
            raise ValueError("Decoded function parameter is not callable.")

        if functionType in (PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF,
                            PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
                            ):
            return data.groupBy(*groupBy).apply(self.pandas_udf)
        if functionType == PythonEvalType.SQL_SCALAR_PANDAS_UDF:
            return data.withColumn(outputCol, self.pandas_udf(inputCol))
        else:
            return data


class PandasUDFScalarTransformer(PandasUDFTransformer, HasInputCol, HasOutputCol,
                           DefaultParamsReadable, DefaultParamsWritable):
    """Allows PySpark User-Defined Function to be used in a Pipeline.

    Parameters
    ----------
    function:
        must be a callable object which can be pickled, or a string representation thereof obtained with
        HasFunction.encode_function.
    functionType:
        must be a value defined by the enumeration PythonEvalType.
    returnType:
        DDL-formatted string describing the schema of the returned Spark dataframe.
    """
    @keyword_only
    def __init__(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, inputCol="", outputCol=""):
        super(Transformer, self).__init__()

        self._setDefault(function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, inputCol="", outputCol="")

        kwargs = self._input_kwargs

        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, inputCol="", outputCol=""):
        """
        setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, inputCol="", outputCol="")
        """
        kwargs = self._input_kwargs

        # encoding is also requested in HasFunction.setFunction, but must also be done here or the TypeConverter
        # will complain
        kwargs['function'] = self.encode_function(function)

        self._set(**kwargs)

        return self

    def _transform(self, data):
        function = self.getFunction()
        functionType = self.getFunctionType()
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        f = self.decode_function(function)

        if not callable(f):
            raise ValueError("Decoded function parameter is not callable.")

        if functionType == PythonEvalType.SQL_SCALAR_PANDAS_UDF:
            return data.withColumn(outputCol, self.pandas_udf(inputCol))
        else:
            return data

if __name__ == "__main__":
    import sys
    if sys.version > '3':
        basestring = str
        xrange = range
        unicode = str

    from pyspark.sql.types import IntegerType, StringType
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .master("local") \
        .appName("FlatTestTime") \
        .config("spark.sql.execution.arrow.enabled", True) \
        .getOrCreate()

    # todo: import this from pyspark.sql.functions ... except WINDOW is missing ???
    class PandasUDFType(object):
        """Pandas UDF Types. See :meth:`pyspark.sql.functions.pandas_udf`.
        """
        SCALAR = PythonEvalType.SQL_SCALAR_PANDAS_UDF
        GROUPED_MAP = PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF
        GROUPED_AGG = PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF

    foo = """
    # from pyspark.sql.functions import pandas_udf, PandasUDFType
    df = spark.createDataFrame((1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
        ("id", "v"))
    @pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)
    def normalize(pdf):
        v = pdf.v
        return pdf.assign(v=(v - v.mean()) / v.std())
        
    df.groupby("id").apply(normalize).show()
    """
    df = spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], ("id", "v"))

    #@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)
    def normalize(pdf):
        v = pdf.v
        return pdf.assign(v=(v - v.mean()) / v.std())

    schema = "id long, v double"
    fubb = PandasUDFGroupedTransformer(function=normalize,
                                       functionType=PandasUDFType.GROUPED_MAP,
                                       returnType=schema,
                                       groupBy=["id"],
                                       )
    pd_normalize = fubb.pandas_udf
    dfubb = df.groupby("id").apply(pd_normalize)

    dfubb.show()

    dfubb2 = fubb.transform(df)
    dfubb2.show()
    dfubb2.printSchema()

    #slen = lambda s: s.str.len()
    def slen(s):
        return s.str.len()

    def to_upper(s):
        return s.str.upper()

    def add_one(x):
        return x + 1


    fubbslen = PandasUDFScalarTransformer(function=slen, returnType="int", inputCol="name", outputCol="name_length")
    fubbupper = PandasUDFScalarTransformer(function=to_upper, returnType="string", inputCol="name", outputCol="name_upper")
    fubbaddone = PandasUDFScalarTransformer(function=add_one, returnType="integer",  inputCol="age", outputCol="new_age")

    dfscalar = spark.createDataFrame([(1, "John Doe", 21)],
                                     ("id", "name", "age"))

    dfslen = fubbslen.transform(dfscalar)
    dfupper = fubbupper.transform(dfslen)
    dfaddone = fubbaddone.transform(dfupper)

    dfaddone.show()

    def normalizer(pdf):
        v = pdf["age"]
        pdf["age_resid"] = (v - v.mean()) / v.std()
        return pdf

    fubr = PandasUDFGroupedTransformer(function=normalizer,
                                       functionType=PandasUDFType.GROUPED_MAP,
                                       returnType="id long, name string, age INT, age_resid float",
                                       groupBy=["id"],
                                       )

    dfscalar1 = spark.createDataFrame([(1, "John Doe", 21),
                                       (1, "Jane Doe", 11),
                                       (2, "John Roe", 31)],
                                      ("id", "name", "age"))

    pipe = Pipeline(stages=[fubr, fubbslen, fubbupper, fubbaddone])
    pipemod = pipe.fit(dfscalar1)

    dfubb3 = pipemod.transform(dfscalar1)
    dfubb3.show()
    dfubb3.printSchema()

    def mean(pdf):
        v = pdf["age"]
        pdf["age_mean"] = v.mean()
        return pdf

    gagg = PandasUDFGroupedTransformer(function=mean,
                                       functionType=PandasUDFType.GROUPED_AGG,
                                       returnType="id long, age_meam double",
                                       groupBy=["id"],
                                       )

    gagg.transform(dfscalar1).show()


    foo_scalar = """
       from pyspark.sql.functions import pandas_udf, PandasUDFType
       from pyspark.sql.types import IntegerType, StringType
       slen = pandas_udf(lambda s: s.str.len(), IntegerType())
       @pandas_udf(StringType())
       def to_upper(s):
           return s.str.upper()
       @pandas_udf("integer", PandasUDFType.SCALAR)
       def add_one(x):
           return x + 1
       df = spark.createDataFrame([(1, "John Doe", 21)],
                                  ("id", "name", "age"))
       df.select(slen("name").alias("slen(name)"), to_upper("name"), add_one("age")) \
           .show()
    """
    # ToDo: separate scalar and grouped_map/grouped_agg
    # scalar should accept inputCol, outputCol,
    # just use data.withColumn(outputCol, self.pandas_udf(inputCol))

    # def fungus(pdf):
    #     return 2 * pdf["vorpal"]
    # baz = PandasUDFTransformer(function=fungus, functionType=200, returnType="INT")
    # print(baz.getFunction())
    # print(baz.getReturnType())
    # print(baz.getFunctionType())
    #
    # dfun = baz.transform(df)
    # dfun.show(5)
    print("Wahoo")