from pyarrow import types as patypes

from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.mllib.linalg import VectorUDT as VectorUDT_mllib

# ToDo: consider stripping error handler from here, since it should never be called except for VectorUDT
# def _getVectorLengthAndType(name, row):
#     num_elements = 0
#     dtype = None
#     ptype = None
#     try:
#         v0 = row[name]
#         if isinstance(v0, DenseVector):
#             num_elements = len(v0)
#             value = v0[0]
#             if types.is_float_value(value):
#                 dtype = DoubleType()  # VectorDataType.DOUBLE
#                 ptype = float
#             elif types.is_integer_value(value):
#                 dtype = IntegerType()  # VectorDataType.INTEGER
#                 ptype = int
#             elif types.is_string(value):
#                 dtype = StringType()  # VectorDataType.STRING
#                 ptype = str
#             else:
#                 # maybe not the best default choice, but...
#                 print("Unrecognized datatype {}, attempting to use Double".format(type(value)))
#                 dtype = DoubleType()  # VectorDataType.DOUBLE
#                 ptype = float
#     except Exception as exc:
#         print("Skipping VectorUDT {} due to error:\n{}".format(name, exc))
#     return num_elements, dtype, ptype
# 
# 
# def _flatten(name, row):
#     length, dtype, ptype = _getVectorLengthAndType(name, row)
#     elements = []
# 
#     def ith_(v, i):
#         try:
#             return ptype(v[i])
#         except ValueError:
#             return None
# 
#     ith = udf(ith_, dtype)
# 
#     for i in range(length):
#         name_i = "{}[{}]".format(name, i)
#         elements.append(ith(col(name), lit(i)).alias(name_i))
# 
#     # in case something went wrong, pass through unaltered
#     if length == 0:
#         elements.append(name)
# 
#     return elements
# 
# def flattenVectorUDT(data):
#     row0 = data.take(1)[0]
#     found_vector = False
#     new_cols = []
#     for field in data.schema:
#         name = field.name
#         if type(field.dataType) in (VectorUDT, VectorUDT_mllib):
#             new_cols.extend(_flatten(name, row0))
#             found_vector = True
#         else:
#             new_cols.append(col(name))
# 
#     if found_vector:
#         return data.select(new_cols)
#     else:
#         return data

def _getVectorLengthAndType(name, row):
    num_elements = 0
    dtype = DoubleType()
    ptype = float
    isDense = False
    isSparse = False

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

    try:
        v0 = row[name]
        num_elements = len(v0)
        if isinstance(v0, (DenseVector, SparseVector)):
            value = v0[0]
        else:
            print("Not a vector? Default to using DoubleType")
            value = 0.0
        dtype, ptype = _getType(value)
    except Exception as exc:
        print("Skipping VectorUDT {} due to error:\n{}".format(name, exc))
    return num_elements, dtype, ptype

def _flatten(name, length, dtype, ptype):
    elements = []

    def ith_(v, i):
        try:
            return ptype(v[i])
        except ValueError:
            return None

    ith = udf(ith_, dtype)

    for i in range(length):
        name_i = "{}[{}]".format(name, i)
        elements.append(ith(col(name), lit(i)).alias(name_i))

    return elements

# use sql.select to flatten dense vectors
def flattenVectorUDT(data):
    row0 = data.take(1)[0]
    found_vector = False
    vector_cols = []

    for field in data.schema:
        name = field.name
        if type(field.dataType) in (VectorUDT, VectorUDT_mllib):
            found_vector = True
            num_elements, dtype, ptype = _getVectorLengthAndType(name, row0)
            vector_cols.extend(_flatten(name, num_elements, dtype, ptype))
        else:
            vector_cols.append(col(name))

    # convert dense to columns
    if found_vector:
        return data.select(vector_cols)
    else:
        return data