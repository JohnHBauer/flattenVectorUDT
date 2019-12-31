# flattenVectorUDT
Each vector-valued column in a Spark DataFrame is replaced by ordinary columns to facilitate use in PySpark.

Columns of type `VectorUDT`are identified, and each vector element is placed in its own column.
For example, `LogisticRegressionModel` creates a column `probability`, which will be replaced 
by columns `probaility[0]`, `probability[1]`, ... Note that these are not indexed, but instead 
[0] is part of the column's name.

`VectorAssembler`, `OneHotEncoder`, and `Model` objects create User-Defined Types 
which are vector-valued: either `DenseVector` or `SparseVector`, the elements of which may be
floats or integers.

The elements of these vectors are not directly indexable in PySpark.  Moreover, when creating a 
`pandas_udf` using pyarrow, if a VectorUDT is encountered the conversion will not be optimized.
Instead, a pandas Series of Object will result.  These values must be converted row by row 
using .tolist() or .values.  Working with these in pandas is a tedious and  error-prone process.

The vectorFlattenerUDT function provides the core functionality.  Given a dataframe, it detects
columns with dataType `VectorUDT`, inspects one row to determine the length and value type, 
generates a udf which places each vector element in its own column, and uses SparkSQL `select`
to return a dataframe without `VectorUDT`s.

If meta-data for the length and element type of VectorUDT was consistently and reliably set,
there would be no need to inspect a row in order to infer them.

A `Transformer` class minimally wraps the above, and an `Estimator`/`Model` pair.  The `fit` method
finds the names, lengths, and data types of vectors, and makes them available as parameters of the 
`Model` obect which it emits.

A `VectorReAssembler` detects columns whose names end in `[###]` (where ### is a sequence of digits)
and uses `VectorAssembler` to re-assemble them into a vector.  A dataframe which is flattened and 
re-assembled will usually be identical to the initial dataframe.  However, note that any 
`SparseVector`-valued columns will be re-assembled as `DenseVector`.

Ideally, call `flattenVectorUDT` before `pandas_udf`, apply some pandas transformations,
(drop unneeded columns), and re-assemble.  `pandas_udf` could be decorated to do this.

In the long term, Scala code along the lines of projection of each vector element
into its own column followed by flatmap, together with a PySpark wrapper,
might look a little cleaner.