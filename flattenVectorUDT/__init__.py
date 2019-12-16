import sys
import re

if sys.version >= '3':
    basestring = unicode = str
    long = int

from collections import defaultdict

from pyarrow import types as patypes

from pyspark import keyword_only

from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.mllib.linalg import VectorUDT as VectorUDT_mllib
from pyspark.sql.functions import udf, lit, col
from pyspark.sql.types import *


class VectorFlattener(Transformer):
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

        # vectors = self._findVectorUDT(data)
        # new_columns = []
        # data_columns = data.columns
        # # n.b. this will bomb if no data
        # # no way to recover if there isn't even a single row, so don't bother with error handling
        # row0 = data.take(1)[0]
        # for name in data_columns:
        #     if name in vectors:
        #         flattened_cols = self._flatten(name, row0)
        #         new_columns.extend(flattened_cols)
        #     else:
        #         new_columns.append(name)
        #
        # if len(data_columns) == len(new_columns):
        #     return data
        # else:
        #     return data.select(*new_columns)

class VectorReAssembler(Transformer):
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