from pyarrow import types

from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.mllib.linalg import VectorUDT as VectorUDT_mllib


def _getVectorLengthAndType(name, row):
    try:
        v0 = row[name]
        if isinstance(v0, DenseVector):
            num_elements = len(v0)
            value = v0[0]
            if types.is_float_value(value):
                dtype = DoubleType()  # VectorDataType.DOUBLE
                ptype = float
            elif types.is_integer_value(value):
                dtype = IntegerType()  # VectorDataType.INTEGER
                ptype = int
            elif types.is_string(value):
                dtype = StringType()  # VectorDataType.STRING
                ptype = str
            else:
                # maybe not the best default choice, but...
                print("Unrecognized datatype {}, attempting to use Double".format(type(value)))
                dtype = DoubleType()  # VectorDataType.DOUBLE
                ptype = float
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

def flattenVectorUDT(data):
    row0 = data.take(1)[0]
    found_vector = False
    new_cols = []
    for field in data.schema:
        name = field.name
        if type(field.dataType) in (VectorUDT, VectorUDT_mllib):
            num_elements, dtype, ptype = _getVectorLengthAndType(name, row0)
            new_cols.extend(_flatten(name, num_elements, dtype, ptype))
            found_vector = True
        else:
            new_cols.append(col(name))

    if found_vector:
        return data.select(new_cols)
    else:
        return data