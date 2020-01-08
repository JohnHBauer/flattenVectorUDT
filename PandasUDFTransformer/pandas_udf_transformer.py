import codecs
import pickle

from pyspark import keyword_only

from pyspark.ml import Estimator, Model, Transformer, Pipeline
from pyspark.rdd import PythonEvalType
from pyspark.ml.param import Param, Params, TypeConverters
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


class PandasUDFTransformer(Transformer, HasFunction, HasFunctionType, HasReturnType, HasGroupBy,
                           DefaultParamsReadable, DefaultParamsWritable):
    """Allows PySpark User-Defined Function to be used in a Pipeline.

    Parameters
    ----------
    function:
        must be a callable object which can be pickled, or a string representation thereof obtained with encode_function.
    functionType:
        must be a value defined by the enumeration PythonEvalType.
    returnType:
        DDL-formatted string describing the schema of the returned Spark dataframe.
    """
    @keyword_only
    def __init__(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[]):
        super(Transformer, self).__init__()

        self._setDefault(function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[])

        kwargs = self._input_kwargs

        # if callable(function):
        #     f = function
        #      # make a string suitable for read/write
        #     function = self.encode_function(function)
        #     #function = codecs.encode(pickle.dumps(function), "base64").decode()
        #     kwargs['function'] = function
        # elif isinstance(function, basestring):
        #     f = self.encode_function(function)
        #     #f = pickle.loads(codecs.decode(function.encode(), "base64"))

        self.setParams(**kwargs)

        # self._pandas_udf = pandas_udf(f=f,
        #                               returnType=returnType,
        #                               functionType=functionType,
        #                               )


    @keyword_only
    def setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[]):
        """
        setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[])
        """
        kwargs = self._input_kwargs

        # if callable(function):
        #      # make a string suitable for read/write so Transformer can be saved and loaded
        #     function = codecs.encode(pickle.dumps(function), "base64").decode()
        kwargs['function'] = self.encode_function(function)

        self._set(**kwargs)

        return self

    def _col_type_name(self):
        # really shouldn't worry about/allow comma
        tokens = self.getReturnType().split(",")[0].split()
        # consider: should this  strict, i.e. len(tokens) == 2 ?
        if len(tokens) > 1:
            col_type = tokens[0]
            col_name = tokens[1]
        elif tokens:
            col_type = tokens[0]
            col_name = ""
        else:
            col_type = "string"
            col_name = ""
        return col_type, col_name

    def _transform(self, data):
        function = self.getFunction()
        returnType = self.getReturnType()
        functionType = self.getFunctionType()
        groupBy = self.getGroupBy()

        #f = pickle.loads(codecs.decode(function.encode(), "base64"))
        f = self.decode_function(function)
        
        if not callable(f):
            raise ValueError("Decoded function parameter is not callable.")

        if functionType == PythonEvalType.SQL_SCALAR_PANDAS_UDF:
            col_type, col_name = self._col_type_name()
            if col_name:
                return data.withColumn(col_name, self.pandas_udf())
            # n.b. comparable to data.select(self.pandas_udf).alias(col_name)
            else:
                # column name will be constructed by sql: function(col)
                return data.select(self.pandas_udf())
        elif functionType in (PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF,
                            PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
                            ):
            return data.groupBy(*groupBy).apply(self.pandas_udf())
        else:
            return data

        #
        # pdudf = pandas_udf(f=f,
        #                    returnType=returnType,
        #                    functionType=functionType,
        #                    )
        # return pdudf(data)

    def pandas_udf(self):
        function = self.getFunction()
        returnType = self.getReturnType()
        functionType = self.getFunctionType()
        #groupBy = self.getGroupBy()

        # f = pickle.loads(codecs.decode(function.encode(), "base64"))
        f = self.decode_function(function)

        if not callable(f):
            raise ValueError("Decoded function parameter is not callable.")

        col_type, _ = self._col_type_name()

        return pandas_udf(f=f,
                          returnType=returnType,
                          functionType=functionType,
                          )

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
            raise ValueError("decoded string must be a function")
        # could check for 1 or 2 positional arguments ???
        return f

class PandasUDFGroupedTransformer(PandasUDFTransformer,  HasGroupBy,
                           DefaultParamsReadable, DefaultParamsWritable):
    """Allows PySpark User-Defined Function to be used in a Pipeline.

    Parameters
    ----------
    function:
        must be a callable object which can be pickled, or a string representation thereof obtained with encode_function.
    functionType:
        must be a value defined by the enumeration PythonEvalType.
    returnType:
        DDL-formatted string describing the schema of the returned Spark dataframe.
    """
    @keyword_only
    def __init__(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[]):
        super(Transformer, self).__init__()

        self._setDefault(function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[])

        kwargs = self._input_kwargs

        # if callable(function):
        #     f = function
        #      # make a string suitable for read/write
        #     function = self.encode_function(function)
        #     #function = codecs.encode(pickle.dumps(function), "base64").decode()
        #     kwargs['function'] = function
        # elif isinstance(function, basestring):
        #     f = self.encode_function(function)
        #     #f = pickle.loads(codecs.decode(function.encode(), "base64"))

        self.setParams(**kwargs)

        # self._pandas_udf = pandas_udf(f=f,
        #                               returnType=returnType,
        #                               functionType=functionType,
        #                               )


    @keyword_only
    def setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[]):
        """
        setParams(self, function="", returnType="", functionType=PythonEvalType.SQL_SCALAR_PANDAS_UDF, groupBy=[])
        """
        kwargs = self._input_kwargs

        # if callable(function):
        #      # make a string suitable for read/write so Transformer can be saved and loaded
        #     function = codecs.encode(pickle.dumps(function), "base64").decode()
        kwargs['function'] = self.encode_function(function)

        self._set(**kwargs)

        return self

if __name__ == "__main__":
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
    fubb = PandasUDFTransformer(function=normalize,
                                functionType=PandasUDFType.GROUPED_MAP,
                                returnType=schema,
                                groupBy=["id"])
    pd_normalize = fubb.pandas_udf()
    dfubb = df.groupby("id").apply(pd_normalize)

    dfubb.show()

    dfubb2 = fubb.transform(df)
    dfubb2.show()

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