import pandas
import numpy
import pickle
import glob
import os
import datetime
from IPython.core.display import display, HTML


def pickle_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def pickle_load(file):
    with open(file, 'rb') as f:
        rv = pickle.load(f)
    return rv


def df_to_pickle(input_df, output_dir, file_prefix, chunksize=50000):
    chunks = len(input_df) // chunksize + 1
    for i in range(chunks):
        input_df[i * chunksize:(i + 1) * chunksize].to_pickle(os.path.normpath(os.path.join(output_dir, file_prefix + "_p" + str(i).zfill(4) + ".pkl")))


def df_from_pickle(input_dir, file_prefix):
    files = glob.glob(os.path.normpath(os.path.join(input_dir, file_prefix + "_p*.pkl")))
    files.sort()
    dflist = list()
    for f in files:
        dflist.append(pandas.read_pickle(f))
    return pandas.concat(dflist, axis=0)


def df_contents(input_df: pandas.DataFrame,
                display_html_output: bool = False,
                excel_filename: str = None,
                excel_sheet_name: str = None
                ):
    """        
    :type input_df: pandas.DataFrame
    :type display_html_output: bool
    :type excel_filename: str
    :type excel_sheet_name: str
    """
    columns = input_df.columns.sort_values().tolist()
    dflength = len(input_df)
    rdtype = list()
    rn = list()
    rnmiss = list()
    runique = list()
    rint = list()
    rfloat = list()
    rboolean = list()
    rnominal = list()
    rdatetime = list()
    rtimedelta = list()
    rmin = list()
    rmax = list()
    rmean = list()
    rstd = list()
    rmode = list()
    rmed = list()
    rp1 = list()
    rp5 = list()
    rp10 = list()
    rp25 = list()
    rp75 = list()
    rp90 = list()
    rp95 = list()
    rp99 = list()
    for col in columns:
        _dtype = input_df[col].dtype
        if _dtype == numpy.dtype('float64'):
            rdtype.append("float64")
        elif _dtype == numpy.dtype('float32'):
            rdtype.append("float32")
        elif _dtype == numpy.dtype('int64'):
            rdtype.append("int64")
        elif _dtype == numpy.dtype('int32'):
            rdtype.append("int32")
        elif _dtype == numpy.dtype('O'):
            rdtype.append("object")
        elif _dtype == numpy.dtype('<M8[ns]'):
            rdtype.append("datetime64[ns]")
        elif _dtype == numpy.dtype('<m8[ns]'):
            rdtype.append("timedelta64[ns]")
        else:
            rdtype.append("")

        _count = input_df[col].count()
        rn.append(_count)
        rnmiss.append(dflength - _count)
        runique.append(len(input_df[col].unique()))

        try:
            rmean.append(input_df[col].mean())
        except:
            rmean.append(numpy.NaN)

        try:
            rmode.append(input_df[col].mode().tolist())
        except:
            rmode.append(numpy.NaN)

        try:
            rstd.append(input_df[col].std())
        except:
            rstd.append(numpy.NaN)

        try:
            rmin.append(input_df[col].min())
        except:
            rmin.append(numpy.NaN)

        try:
            rmax.append(input_df[col].max())
        except:
            rmax.append(numpy.NaN)

        try:
            q = input_df[col].quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).tolist()
        except:
            q = [numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN]

        rp1.append(q[0])
        rp5.append(q[1])
        rp10.append(q[2])
        rp25.append(q[3])
        rmed.append(q[4])
        rp75.append(q[5])
        rp90.append(q[6])
        rp95.append(q[7])
        rp99.append(q[8])

        _t = input_df[col][pandas.notnull(input_df[col])].apply(type)
        _dtypes = _t.groupby(_t).count()
        for i in range(0, len(_dtypes)):
            rboolean.append(_dtypes[i] if _dtypes.index[i] is bool else 0)
            rfloat.append(_dtypes[i] if _dtypes.index[i] is float else 0)
            rint.append(_dtypes[i] if _dtypes.index[i] is int else 0)
            rnominal.append(_dtypes[i] if _dtypes.index[i] is str else 0)
            rdatetime.append(_dtypes[i] if _dtypes.index[i] is pandas.Timestamp else 0)
            rtimedelta.append(_dtypes[i] if _dtypes.index[i] is pandas.Timedelta else 0)

    rdf = pandas.DataFrame(numpy.column_stack([rdtype, rnmiss, rn, rboolean, rfloat,
                                               rint, rnominal, rdatetime, rtimedelta, runique, rmode, rmean, rstd,
                                               rmin, rp1, rp5, rp10, rp25, rmed, rp75, rp90, rp95, rp99, rmax]),
                           columns=["column_dtype", "n_missing", "n_non_missing", "n_boolean", "n_float",
                                    "n_integer", "n_string", "n_datetime", "n_timedelta",
                                    "n_unique_values", "mode", "mean", "stdev", "min", "p1", "p5", "p10", "p25", "median",
                                    "p75", "p90", "p95", "p99", "max"],
                           index=columns
                           )
    rdf.index.name = 'Column Name'

    if excel_filename is not None:
        _sheet = "Sheet1" if excel_sheet_name is None else excel_sheet_name
        _writer = pandas.ExcelWriter(excel_filename)
        rdf.to_excel(_writer, _sheet)
        _writer.save()

    if display_html_output:
        _html = rdf.to_html(justify="right", notebook=True)
        display(HTML(_html))

    return rdf
