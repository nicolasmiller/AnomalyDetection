from datetime import datetime
from re import match
from heapq import nlargest
import pytz

def datetimes_from_ts(column):
    return column.map(
        lambda datestring: datetime.fromtimestamp(int(datestring), tz=pytz.utc))

def date_format(column, format):
    return column.map(lambda datestring: strptime(datestring, format))

def format_timestamp(indf, index=0):
    if indf.dtypes[0].type is np.datetime64:
        return indf

    column = indf.iloc[:,index]

    if match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} \\+\\d{4}$",
             column[0]):
        column = date_format(column, "%Y-%m-%d %H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$", column[0]):
        column = date_format(column, "%Y-%m-%d %H:%M:%S")
    elif match("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}$", column[0]):
        column = date_format(column, "%Y-%m-%d %H:%M")
    elif match("^\\d{2}/\\d{2}/\\d{2}$", column[0]):
        column = date_format(column, "%m/%d/%y")
    elif match("^\\d{2}/\\d{2}/\\d{4}$", column[0]):
        column = date_format(column, "%Y%m%d")
    elif match("^\\d{4}\\d{2}\\d{2}$", column[0]):
        column = date_format(column, "%Y/%m/%d/%H")
    elif match("^\\d{10}$", column[0]):
        column = datetimes_from_ts(column)

    return indf

def get_gran(tsdf, index=0):
    col = indf.iloc[:,index]
    n = len(col)

    largest, second_largest = nlargest(2, col)
    gran = int(round((largest - second_largest) / np.timedelta64(1, 's')))

    if gran >= 86400:
        return "day"
    elif gran >= 3600:
        return "hr"
    elif gran >= 60:
        return "min"
    elif gran >= 1:
        return "sec"
    else:
        return "ms"
