import re

from collections import defaultdict, namedtuple
from enum import IntEnum
from pyarrow import types as patypes
from pyspark import keyword_only

from pyspark.ml import Estimator, Model, Transformer, Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.mllib.linalg import VectorUDT as VectorUDT_mllib
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, lit, col
from pyspark.sql.types import DoubleType, IntegerType, StringType


class VectorFlattener(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """Transform columns of type VectorUDT from Series of DenseVector into separate columns.

     Spark models produce columns such as prediction, probability which are vector-valued.
     Their input is always a features vector.  PySpark cannot access the elements of such vectors.
     Moreover, the presence of these columns prevents optimized conversion to pandas using pyarrow,
     and results in a pandas Series with DenseVector elements, which are awkward to work with in pandas.

     Column names containing `[###]` at the end (where ### is any sequence of digits) should be avoided by convention.

    For example, if `probability` contains vectors of length 2, they will become columns `probability[0]`
    and `probability[1]`.
    todo: include example output using dummy table probability => probability[0], probability[1]
    """
    # n.b. this version flattens all vectors at once

    @staticmethod
    def _findVectorUDT(data):
        names = []
        for field in data.schema:
            if type(field.dataType) in (VectorUDT, VectorUDT_mllib):
                name = field.name
                names.append(name)
        return names

    @staticmethod
    def _getType(value):
        if patypes.is_float_value(value):
            dtype = DoubleType()  # VectorDataType.DOUBLE
            ptype = float
        elif patypes.is_integer_value(value):
            dtype = IntegerType()  # VectorDataType.INTEGER
            ptype = int
        elif patypes.is_string(value):
            dtype = StringType()  # VectorDataType.STRING
            ptype = str
        else:
            # maybe not the best default choice, but...
            print("Unrecognized datatype {}, attempting to use Double".format(type(value)))
            dtype = DoubleType()  # VectorDataType.DOUBLE
            ptype = float
        return dtype, ptype

    @staticmethod
    def _flatten(name, row):
        elements = []
        try:
            v0 = row[name]
            if isinstance(v0, (DenseVector, SparseVector)):
                num_elements = len(v0)
                value = v0[0]
                dtype, ptype = VectorFlattener._getType(value)

                # function to retrieve element i of a VectorUDT using SparkSQL
                get_element = udf(lambda v, i: ptype(v[int(i)]), dtype)

                for i in range(num_elements):
                    name_i = "{}[{}]".format(name, i)
                    elements.append(get_element(name, lit(i)).alias(name_i))
        except Exception as exc:
            print("Skipping VectorUDT {} due to error:\n{}".format(name, exc))

        return elements

    @staticmethod
    def flattenVectorUDT(data):
        """Vector output columns are transformed into flattened column space  Other columns are unchanged.

        For example, ML models have a vector-valued column `probability`, which will be replaced by columns
        probability[0], probabbilityt[1], ... with one column for each element of the vector.
        Bug/deficiency in PySpark API makes accessing vector elements by index problematic.
        """
        vectors = VectorFlattener._findVectorUDT(data)
        new_columns = []
        data_columns = data.columns
        # n.b. this will bomb if no data
        # no way to recover if there isn't even a single row, so don't bother with error handling
        row0 = data.take(1)[0]
        for name in data_columns:
            if name in vectors:
                flattened_cols = VectorFlattener._flatten(name, row0)
                new_columns.extend(flattened_cols)
            else:
                new_columns.append(name)

        if len(data_columns) == len(new_columns):
            return data
        else:
            return data.select(*new_columns)

    # called by super transform, any extra params have already been copied to object
    def _transform(self, data):
        """Vector output columns are transformed into flattened column space  Other columns are unchanged.

        For example, ML models have a vector-valued column `probability`, which will be replaced by columns
        probability[0], probabbilityt[1], ... with one column for each element of the vector.
        Bug/deficiency in PySpark API makes accessing vector elements by index problematic.
        """
        return VectorFlattener.flattenVectorUDT(data)

class VectorReAssembler(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """Transform columns produced by VectorFlattener into a Spark VectorUDT.

    Columns of the form {name}[{index}] will be combined into a single vector.
    For example, if a dataframe has columns probablility[0], probability[1] they will be
    replaced by a single vector named probablility, with each row being a DenseVector of length 2.
    That is, it is the inverse transformation of VectorFlattener.

    Columnn names ending in `[###]` should be avoided by convention.
    """
    # class variable to avoid recompiling pattern
    pattern = re.compile(r"([\w]+)\[(\d+)\]$")

    @staticmethod
    def _findVectorElements(data):
        # construct dict whose keys are names of reassembled vectors, values are list of input variables
        max_index = defaultdict(int)
        select_cols = []

        for field in data.schema:
            name = field.name
            match = VectorReAssembler.pattern.match(name)
            if match:
                name_root = match.group(1)
                idx = match.group(2)
                if name_root not in max_index:
                    select_cols.append(name_root)
                max_index[name_root] = max(int(idx), max_index[name_root])
            else:
                select_cols.append(name)
        # ensure there are no gaps in the sequence, names are called in order
        # assume Python defauldict returns keys in correct order ...
        # this *WILL* provoke an error in _transform if any vector elements are missing

        elements = defaultdict(list)
        for name_root, idx in max_index.items():
            #if idx:
            elements[name_root].extend("{}[{}]".format(name_root, i) for i in range(idx + 1))
            #else:
            #    elements[name_root].append(name_root)

        drop_cols = set(name for names in elements.values() for name in names)

        return elements, select_cols, drop_cols
        #return ["{}[{}]".format(name_root, i) for name_root, idx in elements.items() for i in range(idx+1)]


    @staticmethod
    def reassembleVectorUDT(data):
        """Flattened columns from VectorFlattener are reassembled into Vector-valued columns.

        For example, probability[0], probability[1] become probability, with each row being a DenseVector of length 2.
        The schema order is not preserved: reassembled columns will appear after all others.
        Note that if one or more columns with the largest index/indices are dropped, the re-assembled vectors will not
        have the same length as the original.
        Also, note that if a SparseVector column is flattened, it will be reassembled into a DenseVector.
        """
        elements, select_cols, drop_cols = VectorReAssembler._findVectorElements(data)

        assemblers = [VectorAssembler(inputCols=cols, outputCol=name) for name, cols in elements.items()]

        return Pipeline(stages=assemblers)\
            .fit(data)\
            .transform(data)\
            .select(*select_cols)\
            .drop(*list(drop_cols))

    # called by super transform, any extra params have already been copied to object
    # ToDo: preserve schema order by adding a select statement
    def _transform(self, data):
        """Flattened columns from VectorFlattener are reassembled into Vector-valued columns.

        For example, probability[0], probability[1] become probability, with each row being a DenseVector of length 2.
        The schema order is not preserved: reassembled columns will appear after all others.
        Note that if one or more columns with the largest index/indices are dropped, the re-assembled vectors will not
        have the same length as the original.
        Also, note that if a SparseVector column is flattened, it will be reassembled into a DenseVector.
        """
        return VectorReAssembler.reassembleVectorUDT(data)


class VectorDataType(IntEnum):
    """Used to pass dataTypes as listInt from Estimator to Model."""
    DOUBLE = 1
    INTEGER = 2
    STRING = 3

    def toSpark(self):
        value = _SPARK_TYPE.get(self.value, None)
        if value:
            return value
        else:
            raise ValueError("Invalid data type for VectorUDT: {}".format(self.value))

    def toPython(self):
        value = _PYTHON_TYPE.get(self.value, None)
        if value:
            return value
        else:
            raise ValueError("Invalid data type for VectorUDT: {}".format(self.value))

_SPARK_TYPE = {VectorDataType.DOUBLE: DoubleType(),
              VectorDataType.INTEGER: IntegerType(),
              VectorDataType.STRING: StringType(),
              }

# n.b. the pythonType values must be callable
_PYTHON_TYPE = {VectorDataType.DOUBLE: float,
               VectorDataType.INTEGER: int,
               VectorDataType.STRING: str,
               }

class HasLengths(Params):
    """
    Mixin for param lengths: list of the number of elements in each vector.
    """
    lengths = Param(Params._dummy(), "lengths", "Number of elements of each VectorUDT.",
                        typeConverter=TypeConverters.toListInt)

    def __init__(self):
        super(HasLengths, self).__init__()

    def setLengths(self, value):
        """
        Sets the value of :py:attr:`lengths`.
        """
        return self._set(lengths=value)

    def getLengths(self):
        """
        Gets the value of separator or its default value.
        """
        return self.getOrDefault(self.lengths)

class HasDataTypes(Params):
    """
    Mixin for param separator: data type of each vector.
    """
    dataTypes = Param(Params._dummy(), "dataTypes", "Data Type of each VectorUDT.",
                      typeConverter=TypeConverters.toListInt)

    def __init__(self):
        super(HasDataTypes, self).__init__()

    def setDataTypes(self, value):
        """
        Sets the value of :py:attr:`dataType`.
        """
        return self._set(dataTypes=value)

    def getDataTypes(self):
        """
        Gets the value of dataType or its default value.
        """
        return self.getOrDefault(self.dataTypes)


VectorInfo = namedtuple("VectorInfo", ["name", "length", "dataType"])


class VectorFlattenerEstimator(Estimator,
                               DefaultParamsReadable,
                               DefaultParamsWritable,
                               ):
    """Transform columns of type VectorUDT from Series of DenseVector into separate columns.

    Spark models produce columns such as prediction, probability which are vector-valued.
    Their input is always a features vector.  PySpark cannot access the elements of such vectors.
    Moreover, the presence of these columns prevents optimized conversion to pandas using pyarrow,
    and results in a pandas Series with DenseVector elements, which are awkward to work with in pandas.
    For example, if probability contains vectors of length 2, it will be replaced by two columns:
    probability[0] and probability[1].
    Column names ending in `[###]` (where ### is any sequence of digits) should be avoided by convention.

    The estimator inspects a single vector to infer length and type, so that the names of the new columns
    will be available in the model object, and to subsequent pipeline objects.
    It might be preferred to use the dataframe's meta-data for this purpose, but it is not consistently set
    at present.  (Compare VectorAssembler, OneHotEncoder, ... )
    """
    # todo: include example output using dummy table probability => probability[0], probability[1]

    # n.b. this version  flattens all vectors at once
    @keyword_only
    def __init__(self):
        super(VectorFlattenerEstimator, self).__init__()

        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        """
        setParams(self)
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def _findVectorUDT(self, data):
        names = []
        for field in data.schema:
            if type(field.dataType) in (VectorUDT, VectorUDT_mllib):
                name = field.name
                names.append(name)
        return names

    # ToDo: add all the other data types
    # ToDo: use a class or Enum to connect Vector types to sql types via integer look-up,
    #  since they will be passed through a list of integers to insure DefaultParamsReadable, ...Writable
    def _getVectorLengthAndType(self, name, row):
        dtype = VectorDataType.DOUBLE
        try:
            v0 = row[name]
            if isinstance(v0, (DenseVector, SparseVector)):
                num_elements = len(v0)
                value = v0[0]
                if patypes.is_float_value(value):
                    dtype = VectorDataType.DOUBLE  # DoubleType()
                elif patypes.is_integer_value(value):
                    dtype = VectorDataType.INTEGER  # IntegerType()
                elif patypes.is_string(value):
                    dtype = VectorDataType.STRING  # StringType()
                else:
                    # maybe not the best default choice, but...
                    print("Unrecognized datatype {}, attempting to use Double".format(type(value)))
                    dtype = VectorDataType.DOUBLE  # DoubleType()
        except Exception as exc:
            print("Skipping VectorUDT `{}` due to error:\n{}".format(name, exc))
        return num_elements, dtype

    # called by super transform, any extra params have already been copied to object
    def _fit(self, data):
        """Vector output columns are transformed into flattened column space.

        probability[0] becomes probability__0 etc. for each of the n_label index values.
        Bug/deficiency in PySpark API makes accessing vector elements by index problematic.
        """
        vectors = self._findVectorUDT(data)
        lengths = []
        dataTypes = []

        for name in vectors:
            row0 = data.take(1)[0]
            length, dtype = self._getVectorLengthAndType(name, row0)
            lengths.append(length)
            dataTypes.append(dtype.value)
            # outputCols.extend(["{}[{}]".format(name, i) for i in range(length)])
        return VectorFlattenerModel(inputCols=vectors, outputCols=[], lengths=lengths, dataTypes=dataTypes)

class VectorFlattenerModel(Model,
                           HasInputCols,
                           HasOutputCols,
                           HasLengths,
                           HasDataTypes,
                           DefaultParamsReadable,
                           DefaultParamsWritable,
                           ):
    """Transform columns of type VectorUDT from Series of DenseVector into separate columns.

     Spark models produce columns such as prediction, probability which are vector-valued.
     Their input is always a features vector.  PySpark cannot access the elements of such vectors.
     Moreover, the presence of these columns prevents optimized conversion to pandas using pyarrow,
     and results in a pandas Series with DenseVector elements, which are awkward to work with in pandas.

     Columnn names containing the separator (__ by default) should be avoided by convention.

    Parameter:
    ----------
    separator: String to use to generate new column names.  (Default: "__")
    For example, if probability contains vectors of length 2, they will become columns probability__0
    and probability__1.
    todo: include example output using dummy table probability => probability__0, probability__1
    """

    # n.b. this version  flattens all vectors at once
    @keyword_only
    def __init__(self, inputCols=[], outputCols=[], lengths=[], dataTypes=[]):
        super(VectorFlattenerModel, self).__init__()

        self._setDefault(inputCols=[], outputCols=[],  lengths=[], dataTypes=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self.outputColumnMap = defaultdict(list)
        # dict of vectors, key=name, value=VectorInfo(name, length, dataType)
        self.vectorInfo = self._vectorInfo()
        # replace outputCols with names which will be used by transform
        self._setOutputCols()

    @keyword_only
    def setParams(self, inputCols=[], outputCols=[],  lengths=[], dataTypes=[]):
        """
        setParams(self, inputCols=[], outputCols=[],  lengths=[], dataTypes=[])
        """
        kwargs = self._input_kwargs
        self._set(**kwargs)
        return self

    def _setOutputCols(self):
        outputCols = []

        for name in self.getInputCols():
            vector_info = self.vectorInfo[name]
            columns = ["{}[{}]".format(name, i) for i in range(vector_info.length)]
            outputCols.extend(columns)
            self.outputColumnMap[name].extend(columns)

        self.setOutputCols(outputCols)

    def _vectorInfo(self):
        lengths = self.getLengths()
        inputCols = self.getInputCols()
        dataTypes =self.getDataTypes()

        if len(inputCols) == len(lengths) == len(dataTypes):
            info = dict()
            for name, length, dataType in zip(inputCols, lengths, dataTypes):
                info[name] = VectorInfo(name=name, length=length, dataType=VectorDataType(dataType))
            return info
        else:
            raise ValueError("Inconsistency among lists of VectorUDT information.")

    def _flatten(self, name):
        elements = []
        vector_info = self.vectorInfo[name]

        name = vector_info.name
        dtype = vector_info.dataType.toSpark()
        pytype = vector_info.dataType.toPython()

        # function to retrieve element i of a VectorUDT using SparkSQL
        get_element = udf(lambda v, i: pytype(v[int(i)]), dtype)

        for i in range(vector_info.length):
            name_i = "{}[{}]".format(name, i)
            elements.append(get_element(col(name), lit(i)).alias(name_i))
        return elements

    # called by super transform, any extra params have already been copied to object
    def _transform(self, data):
        """Vector-valued output columns are transformed into multi[ple columns.

        The first element of probability becomes probability[0] etc. for each index value.
        Bug/deficiency in PySpark API makes accessing vector elements by index problematic.
        """
        selectCols = []
        for name in data.columns:
            if name in self.vectorInfo:
                selectCols.extend(self._flatten(name))
            else:
                selectCols.append(col(name))

        if self.vectorInfo:
            return data.select(selectCols)
        else:
            return data
