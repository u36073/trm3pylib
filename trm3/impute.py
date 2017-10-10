import time
import pandas
import pandasql
import numpy
import scipy.stats
import math
import pickle
import glob
import os
import datetime
from typing import Dict, Tuple, List
from IPython.core.display import display, HTML

# Version History
#
# 1.1 - Initial Version


def version():
    return "1.1"


class ImputeRules:
    def __init__(self,
                 imputed_columns=None,
                 label=None,
                 groupby=None,
                 groupby_impute_values=None,
                 create_dt=None,
                 notes=None,
                 rules_list=None,
                 verbose=True
                 ):
        """
        :param imputed_columns:
        :type imputed_columns: list[str]
        :param label:
        :type label: str
        :param groupby:
        :type groupby: list[str]
        :param groupby_impute_values:
        :type groupby_impute_values: list[str]
        :param create_dt:
        :type create_dt: datetime.datetime
        :param notes:
        :type notes: str, None
        :param rules_list:
        :type rules_list: list[obj]
        """

        self.imputed_columns = imputed_columns
        self.label = label
        self.groupby = groupby
        self.groupby_impute_values = groupby_impute_values
        self.create_dt = create_dt
        self.notes = notes
        self.rules_list = rules_list

    # ------------------------------------------------------------------
    # Property:  groupby_impute_values
    # ------------------------------------------------------------------
    @property
    def groupby_impute_values(self):
        return self.__groupby_impute_values

    @groupby_impute_values.setter
    def groupby_impute_values(self, x):
        self.__groupby_impute_values = x

    # ------------------------------------------------------------------
    # Property:  imputed_columns
    # ------------------------------------------------------------------
    @property
    def imputed_columns(self):
        return self.__imputed_columns

    @imputed_columns.setter
    def imputed_columns(self, x):
        self.__imputed_columns = x

    # ------------------------------------------------------------------
    # Property:  label
    # ------------------------------------------------------------------
    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, x):
        self.__label = x

    # ------------------------------------------------------------------
    # Property:  groupby
    # ------------------------------------------------------------------
    @property
    def groupby(self):
        return self.__groupby

    @groupby.setter
    def groupby(self, x):
        self.__groupby = x

    # ------------------------------------------------------------------
    # Property:  create_dt
    # ------------------------------------------------------------------
    @property
    def create_dt(self):
        return self.__create_dt

    @create_dt.setter
    def create_dt(self, x):
        self.__create_dt = x

    # ------------------------------------------------------------------
    # Property:  notes
    # ------------------------------------------------------------------
    @property
    def notes(self):
        return self.__notes

    @notes.setter
    def notes(self, x):
        self.__notes = x

    # ------------------------------------------------------------------
    # Property:  rules_list
    # ------------------------------------------------------------------
    @property
    def rules_list(self):
        return self.__rules_list

    @rules_list.setter
    def rules_list(self, x):
        self.__rules_list = x

    #-----------------------------------------------------------------------
    # Method:  version()
    #-----------------------------------------------------------------------
    def version(self):
        return version()

    #-----------------------------------------------------------------------
    # Method:  python_code()
    #-----------------------------------------------------------------------
    def code(self,
             outfile=None,
             include_columns=None,
             exclude_columns=None,
             indicator_suffix="_mi"
             ):

        """
        :param outfile: (Optional) File to save the code text to.  If it already exists, the code
                                   will be appended to the bottom of the file.
        :type outfile: str
        :param include_columns: (Optional) List of column names to generate code for.
        :type include_columns: list[str]
        :param exclude_columns: (Optional) List of column names to exlude from the generated code.
        :type exclude_columns: list[str]
        :rtype: str
        """

        code = ''

        def al(x):
            nonlocal code
            code += x + "\n"

        def expr_val(x) -> str:
            if type(x) == str:
                return "'" + x + "'"
            else:
                return str(x)

        def comment(indent, x):
            spaces = ' ' * 4 * indent
            al(spaces + '#-----------------------------------------------------------------------')
            if type(x) == str:
                al(spaces + '# ' + x)
            elif type(x) == list:
                for line in x:
                    al(spaces + '# ' + line)
            al(spaces + '#-----------------------------------------------------------------------')

        code = "import pandas\n"
        code = "\n"
        code = "def impute_missing_values(df):\n"

        if len(self.__groupby_impute_values) > 0:
            al('')
            comment(1, ["Function to fill missing for a specific column and combination of",
                        "groupby values."])
            al('    def gb_fillna(gb_fill_df, impute_col, expr, impute_val):')
            al('        filter = gb_fill_df.eval(expr)')
            al('        gb_fill_df.loc[filter, impute_col] = gb_fill_df.loc[filter, impute_col].fillna(value=impute_val, inplace=False)')
            al('')
            comment(1, "Replace any missing values in the groupby variables.")

            for col, val in self.__groupby_impute_values:
                al("    df[%s].fillna(%s, inplace=True))" % (expr_val(col), expr_val(val)))

            al("")

        # -----------------------------------------------------------------------
        # Apply missing values for the non-groupby variables
        # -----------------------------------------------------------------------
        for rule in self.__rules_list:

            column_count = rule[0]
            column = rule[1]

            if include_columns is not None:
                if column not in include_columns:
                    continue

            if exclude_columns is not None:
                if column in exclude_columns:
                    continue

            # print("[%03d] %s" % (column_count, column))

            gbdf = rule[3].reset_index(inplace=False)
            row_number = 0
            expression_list = list()
            new_gb_value_impute = None

            al("")
            comment(1, column + " Imputation")

            # -----------------------------------------------------------------------
            # Impute when there are groupby variables
            # -----------------------------------------------------------------------
            if len(self.__groupby_impute_values) > 0:
                # -----------------------------------------------------------------------
                # Begin:  for row_tuple in gbdf.itertuples():
                # -----------------------------------------------------------------------
                for row_tuple in gbdf.itertuples():
                    row = row_tuple._asdict()
                    row_number += 1
                    if row_number == 1:
                        if row["z__column_type"] in ["Numeric"]:
                            new_gb_value_impute = row["z__df_median"]
                        else:
                            new_gb_value_impute = row["z__df_mode"]

                        if row["z__indicator"]:
                            al("    df['%s%s'] = df['%s'].isnull()" % (column, indicator_suffix, column))
                            al("")

                    i = 0
                    expression = ""
                    for col in self.__groupby:
                        i += 1
                        prefix = "" if i == 1 else " and "
                        expression += prefix + col + "==" + expr_val(row[col])
                        expression_list.append(expression)
                        al("    gb_fillna(df,")
                        al("              " + expr_val(column) + ",")
                        al("              " + expr_val(expression) + ",")
                        al("              " + expr_val(row["z__impute"]))
                        al("             )")
                        al("")
                # -----------------------------------------------------------------------
                # End:  for row_tuple in gbdf.itertuples():
                # -----------------------------------------------------------------------

                # -----------------------------------------------------------------------
                # Impute missing when new value(s) for groupby variable(s) is found.
                # -----------------------------------------------------------------------
                i = 0
                new_group_by_expression = ""
                for expr in expression_list:
                    i += 1
                    prefix = "not " if i == 1 else " and not "
                    new_group_by_expression += prefix + "(" + expr + ")"

                al("    gb_fillna(df,")
                al("             " + expr_val(column) + ",")
                al("             " + expr_val(new_group_by_expression) + ",")
                al("             " + expr_val(new_gb_value_impute))
                al("             )")
                al("")
                # -----------------------------------------------------------------------
            # Impute when there are not any groupby variables
            # -----------------------------------------------------------------------
            else:
                for row_tuple in gbdf.itertuples():
                    row = row_tuple._asdict()
                    row_number += 1
                    if row == 1:
                        if row["z__indicator"]:
                            al("    df['%s%s'] = df['%s'].isnull()" % (column, indicator_suffix, column))
                        al("    df[" + expr_val(column) + "].fillna(value=" + expr_val(row["z__impute"]) + ", inplace=True")
                        break
                al("")
        # -----------------------------------------------------------------------
        # End:  for rule in self.__rules_list:
        # -----------------------------------------------------------------------

        if outfile is not None:
            try:
                with open(outfile, 'a') as f:
                    f.write(code)
            except:
                raise

        return code
    # -----------------------------------------------------------------------
    # End of Method:  code()
    # -----------------------------------------------------------------------

    #-----------------------------------------------------------------------
    # Method:  impute()
    #-----------------------------------------------------------------------
    def impute(self,
               impute_df=None,
               inplace=False,
               verbose=True,
               include_columns=None,
               exclude_columns=None,
               missing_columns_action="ignore",  # skip, warn, raise
               indicator_suffix="_mi"
               ):

        """
        :type impute_df: pandas.DataFrame
        :type inplace: bool
        :type verbose: bool
        :type missing_columns_action: str
        :param missing_columns_action: Valid Values --> 'skip' 'warn' 'raise'
        """

        df_columns = impute_df.columns.tolist()
        df_columns.sort()

        new_df = None if inplace else impute_df.loc[:, :].copy()   # type: pandas.DataFrame
        new_gb_value_impute = None

        def expr_val(x) -> str:
            if type(x) == str:
                return "'" + x + "'"
            else:
                return str(x)

        def gb_fillna(df, impute_col, expression, impute_val):
            nonlocal df_columns
            if impute_col in df_columns:
                xfilter = df.eval(expression)
                df.loc[xfilter, impute_col] = df.loc[xfilter, impute_col].fillna(value=impute_val, inplace=False)
            else:
                if missing_columns_action == "warn":
                    print("WARNING:  Column %s was not found." % impute_col)
                    print(" ")
                elif missing_columns_action == "raise":
                    raise Exception("Column %s was not found." % impute_col)
                else:
                    pass

        #-----------------------------------------------------------------------
        # Apply missing value imputations to groupby variables
        #-----------------------------------------------------------------------
        if len(self.__groupby_impute_values) > 0:
            for col, val in self.__groupby_impute_values:
                impute_df[col].fillna(val, inplace=True) if inplace else new_df[col].fillna(val, inplace=True)

        #-----------------------------------------------------------------------
        # Apply missing values for the non-groupby variables
        #-----------------------------------------------------------------------
        for rule in self.__rules_list:

            column_count = rule[0]
            column = rule[1]

            if include_columns is not None:
                if column not in include_columns:
                    continue

            if exclude_columns is not None:
                if column in exclude_columns:
                    continue

            if verbose:
                print("[%03d] %s" % (column_count, column))

            gbdf = rule[3].reset_index(inplace=False)
            row_number = 0
            expression_list = list()

            #-----------------------------------------------------------------------
            # Impute when there are groupby variables
            #-----------------------------------------------------------------------
            if len(self.__groupby_impute_values) > 0:
                #-----------------------------------------------------------------------
                # Begin:  for row_tuple in gbdf.itertuples():
                #-----------------------------------------------------------------------
                for row_tuple in gbdf.itertuples():
                    row = row_tuple._asdict()
                    row_number += 1
                    if row_number == 1:
                        if row["z__column_type"] in ["Numeric"]:
                            new_gb_value_impute = row["z__df_median"]
                        else:
                            new_gb_value_impute = row["z__df_mode"]

                        # -----------------------------------------------------------------------
                        # Create missing indicator variables
                        # -----------------------------------------------------------------------
                        if row["z__indicator"]:
                            if inplace:
                                impute_df[column + indicator_suffix] = impute_df[column].isnull()
                            else:
                                new_df[column + indicator_suffix] = new_df[column].isnull()

                    i = 0
                    expression = ""
                    for col in self.__groupby:
                        i += 1
                        prefix = "" if i == 1 else " and "
                        expression += prefix + col + "==" + expr_val(row[col])
                        expression_list.append(expression)
                        gb_fillna(impute_df if inplace else new_df,
                                  column,
                                  expression,
                                  row["z__impute"]
                                  )
                #-----------------------------------------------------------------------
                # End:  for row_tuple in gbdf.itertuples():
                #-----------------------------------------------------------------------

                #-----------------------------------------------------------------------
                # Impute missing when new value(s) for groupby variable(s) is found.
                #-----------------------------------------------------------------------
                i = 0
                new_group_by_expression = ""
                for expr in expression_list:
                    i += 1
                    prefix = "not " if i == 1 else " and not "
                    new_group_by_expression += prefix + "(" + expr + ")"

                gb_fillna(impute_df if inplace else new_df,
                          column,
                          new_group_by_expression,
                          new_gb_value_impute
                          )
            #-----------------------------------------------------------------------
            # Impute when there are not any groupby variables
            #-----------------------------------------------------------------------
            else:
                for row_tuple in gbdf.itertuples():
                    row = row_tuple._asdict()
                    row_number += 1
                    if row == 1:
                        impute_df[column].fillna(value=row["z__impute"], inplace=True)

                        # -----------------------------------------------------------------------
                        # Create missing indicator variables
                        # -----------------------------------------------------------------------
                        if row["z__indicator"]:
                            if inplace:
                                impute_df[column + indicator_suffix] = impute_df[column].isnull()
                            else:
                                new_df[column + indicator_suffix] = new_df[column].isnull()
                        break

        #-----------------------------------------------------------------------
        # End:  for rule in self.__rules_list:
        #-----------------------------------------------------------------------

        if not inplace:
            return new_df
    #-----------------------------------------------------------------------
    # End of Method:  impute()
    #-----------------------------------------------------------------------


class Imputer:
    def __init__(self,
                 train_df=None,
                 test_df=None,
                 columns=None,
                 groupby=None,
                 label=None,
                 method=1,
                 mode_fallback_distinct_values=5,
                 mean_label_min_obs=.01,
                 mean_label_bins=10,
                 nominal_new_category_min_obs=.01,
                 missing_indicators=False,
                 missing_indicators_min_obs=.01,
                 model_x_columns=None,
                 verbose=True
                 ):
        """
        :param train_df: Pandas Dataframe to generate missing imputation rules from.
        :type train_df: pandas.DataFrame
        :param test_df: Only used when [method]='GBM'.  Pandas Dataframe used as the
               test data to control for overfitting.  If not specified, the train_df
               will be randomly split into train and test groups.
        :type test_df: pandas.DataFrame
        :param columns:
        :type columns: list[str]
        :param groupby:
        :type groupby: list[str]
        :param label:
        :type label: str
        :param method: Imputation method to use for each variable specified in [columns]. Options: \n
                       1  mean > mode \n
                       2  median > mode \n
                       3  mode \n
                       4  mean response > mean/mode > \n
                       5  mean response > median/mode \n
        :type method: int
        :param mode_fallback_distinct_values:
        :type mode_fallback_distinct_values: int
        :param mean_label_min_obs:
        :type mean_label_min_obs: float
        :param mean_label_bins:
        :type mean_label_bins: int
        :param nominal_new_category_min_obs:
        :type nominal_new_category_min_obs: float
        :param missing_indicators:
        :type missing_indicators: bool
        :param missing_indicators_min_obs:
        :type missing_indicators_min_obs: float
        :param model_x_columns:
        :type model_x_columns: list[str]
        :param verbose:
        :type verbose: bool
        """

        self.model_x_columns = model_x_columns
        self.missing_indicators_min_obs = missing_indicators_min_obs
        self.missing_indicators = missing_indicators
        self.nominal_new_category_min_obs = nominal_new_category_min_obs
        self.mean_label_bins = mean_label_bins
        self.mean_label_min_obs = mean_label_min_obs
        self.mode_fallback_distinct_values = mode_fallback_distinct_values
        self.method = method
        self.label = label
        self.groupby = groupby
        self.columns = columns
        self.test_df = test_df
        self.train_df = train_df
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Property:  verbose
    # ------------------------------------------------------------------
    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, x):
        self.__verbose = x

    # ------------------------------------------------------------------
    # Property:  model_x_columns
    # ------------------------------------------------------------------
    @property
    def model_x_columns(self):
        return self.__model_x_columns

    @model_x_columns.setter
    def model_x_columns(self, x):
        self.__model_x_columns = x

    # ------------------------------------------------------------------
    # Property:  missing_indicators_min_obs
    # ------------------------------------------------------------------
    @property
    def missing_indicators_min_obs(self):
        return self.__missing_indicators_min_obs

    @missing_indicators_min_obs.setter
    def missing_indicators_min_obs(self, x):
        self.__missing_indicators_min_obs = x

    # ------------------------------------------------------------------
    # Property:  missing_indicators
    # ------------------------------------------------------------------
    @property
    def missing_indicators(self):
        return self.__missing_indicators

    @missing_indicators.setter
    def missing_indicators(self, x):
        self.__missing_indicators = x

    # ------------------------------------------------------------------
    # Property:  nominal_new_category_min_obs
    # ------------------------------------------------------------------
    @property
    def nominal_new_category_min_obs(self):
        return self.__nominal_new_category_min_obs

    @nominal_new_category_min_obs.setter
    def nominal_new_category_min_obs(self, x):
        self.__nominal_new_category_min_obs = x

    # -----------------------------------------------------------------------
    # Property:  mean_label_bins
    # -----------------------------------------------------------------------
    @property
    def mean_label_bins(self):
        return self.__mean_label_bins

    @mean_label_bins.setter
    def mean_label_bins(self, x):
        self.__mean_label_bins = x

    # -----------------------------------------------------------------------
    # Property:  mean_label_min_observations
    # -----------------------------------------------------------------------
    @property
    def mean_label_min_obs(self):
        return self.__mean_label_min_obs

    @mean_label_min_obs.setter
    def mean_label_min_obs(self, x):
        self.__mean_label_min_obs = x

    # -----------------------------------------------------------------------
    # Property:  numeric_mode_threshold
    # -----------------------------------------------------------------------
    @property
    def mode_fallback_distinct_values(self):
        return self.__mode_fallback_distinct_values

    @mode_fallback_distinct_values.setter
    def mode_fallback_distinct_values(self, x):
        self.__mode_fallback_distinct_values = x

    # -----------------------------------------------------------------------
    # Property:  train_df
    # -----------------------------------------------------------------------
    @property
    def train_df(self):
        return self.__train_df

    @train_df.setter
    def train_df(self, x):
        self.__train_df = x

    # -----------------------------------------------------------------------
    # Property:  test_df
    # -----------------------------------------------------------------------
    @property
    def test_df(self):
        return self.__test_df

    @test_df.setter
    def test_df(self, x):
        self.__test_df = x

    # -----------------------------------------------------------------------
    # Property:  columns
    # -----------------------------------------------------------------------
    @property
    def columns(self):
        return self.__columns

    @columns.setter
    def columns(self, x):
        self.__columns = x

    # -----------------------------------------------------------------------
    # Property:  groupby
    # -----------------------------------------------------------------------
    @property
    def groupby(self):
        return self.__groupby

    @groupby.setter
    def groupby(self, x):
        self.__groupby = x

    # -----------------------------------------------------------------------
    # Property:  label
    # -----------------------------------------------------------------------
    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, x):
        self.__label = x

    # -----------------------------------------------------------------------
    # Property:  method
    # -----------------------------------------------------------------------
    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, x):
        self.__method = x

    #-----------------------------------------------------------------------
    # Method:  version()
    #-----------------------------------------------------------------------
    def version(self):
        return version()

    # -----------------------------------------------------------------------
    # Method:  validate_properties()
    # -----------------------------------------------------------------------
    def validate_properties(self):

        errors = {}
        error_count = 0

        def error(parameter, message):
            nonlocal error_count
            nonlocal errors
            error_count += 1
            errors[error_count] = [parameter, message]

        train_df_cols = []
        test_df_cols = []

        # -----------------------------------------------------------------------
        # train_df
        # -----------------------------------------------------------------------
        if type(self.__train_df) != pandas.DataFrame:
            if self.__train_df is None:
                error("train_df", "A pandas.Dataframe object is required.  Nothing specified.")
            else:
                error("train_df", "A pandas.Dataframe object is required.  Object specified is of type " + str(type(self.__train_df)))
        else:
            train_df_cols = self.__train_df.columns.sort_values().tolist()

        # -----------------------------------------------------------------------
        # test_df
        # -----------------------------------------------------------------------
        if self.__test_df is not None and self.__method == 6 and type(self.__test_df) != pandas.DataFrame:
            error("test_df", "Only accepts a pandas.Dataframe object.  Object specified is of type " + str(type(self.__test_df)))
        elif self.__method == 6 and type(self.__test_df) == pandas.DataFrame:
            test_df_cols = self.__test_df.columns.sort_values().tolist()

        # -----------------------------------------------------------------------
        # columns
        # -----------------------------------------------------------------------
        if self.__columns is None:
            error("columns", "A string or list of strings is required.  Nothing specified.")
        elif type(self.__columns) not in [str, list]:
            error("columns", "A string or list of strings is required.  Object specified is of type " + str(type(self.__train_df)))
        elif type(self.__train_df) == pandas.DataFrame:
            if type(self.__columns) == str:
                columns = [self.__columns]
            else:
                columns = self.__columns

            n = -1
            for col in columns:
                n += 1
                if type(col) != str:
                    error("columns", "An element of the list is not a string.  columns[" + str(n) + "] is of type " + str(type(self.__test_df)))
                if col not in train_df_cols:
                    error("columns", "Column " + col + " not found in the pandas.DataFrame specified in the [train_df] parameter.")

        # -----------------------------------------------------------------------
        # groupby
        # -----------------------------------------------------------------------
        if self.__groupby is not None:
            pass
        elif type(self.__groupby) not in [str, list]:
            error("columns", "A string or list of strings is expected.  Object specified is of type " + str(type(self.__train_df)))
        elif type(self.__train_df) == pandas.DataFrame:
            if type(self.__columns) == str:
                columns = [self.__columns]
            else:
                columns = self.__columns

            n = -1
            for col in columns:
                n += 1
                if type(col) != str:
                    error("groupby", "An element of the list is not a string.  columns[" + str(n) + "] is of type " + str(type(self.__test_df)))
                if col not in train_df_cols:
                    error("groupby", "Column " + col + " not found in the pandas.DataFrame specified in the [train_df] parameter.")

        # -----------------------------------------------------------------------
        # label
        # -----------------------------------------------------------------------
        if self.__label is None and self.__method in [1, 2, 3]:
            pass
        elif type(self.__label) != str and self.__method in [4, 5, 6]:
            error("label", "A string is expected.  Object specified is of type " + str(type(self.__train_df)))
        elif self.__label not in train_df_cols and len(train_df_cols) > 0:
            error("label", "Column " + str(self.__label) + " not found in the pandas.DataFrame specified in the [train_df] parameter.")

        # -----------------------------------------------------------------------
        # method
        # -----------------------------------------------------------------------
        if self.__method not in [1, 2, 3, 4, 5, 6]:
            error("method", "Invalid value.  An integer between 1 and 6 is expected.")

        # -----------------------------------------------------------------------
        # mode_fallback_distinct_values
        # -----------------------------------------------------------------------
        if type(self.__mode_fallback_distinct_values) not in [float, int]:
            error("mode_fallback_distinct_values", "A number greater than or equal to 0 is expected. A non-numeric object was specified of type " + str(type(self.__train_df)))
        elif self.__mode_fallback_distinct_values < 0:
            error("mode_fallback_distinct_values", "A number greater than or equal to 0 is expected. A value of " + str(self.__mode_fallback_distinct_values) + "was specified.")

        # -----------------------------------------------------------------------
        # mean_label_min_obs
        # -----------------------------------------------------------------------
        if type(self.__mean_label_min_obs) not in [float, int]:
            error("mean_label_min_obs", "A number greater than or equal to 0 is expected. A non-numeric object was specified of type " + str(type(self.__train_df)))
        elif self.__mean_label_min_obs < 0:
            error("mean_label_min_obs", "A number greater than or equal to 0 is expected. A value of " + str(self.__mode_fallback_distinct_values) + "was specified.")

        # -----------------------------------------------------------------------
        # mean_label_bins
        # -----------------------------------------------------------------------
        if type(self.__mean_label_bins) not in [float, int]:
            error("mean_label_bins", "A number greater than or equal to 2 is expected. A non-numeric object was specified of type " + str(type(self.__train_df)))
        elif self.__mean_label_bins < 2:
            error("mean_label_bins", "A number greater than or equal to 2 is expected. A value of " + str(self.__mode_fallback_distinct_values) + "was specified.")

        # -----------------------------------------------------------------------
        # nominal_new_category_min_obs
        # -----------------------------------------------------------------------
        if type(self.__nominal_new_category_min_obs) not in [float, int]:
            error("nominal_new_category_min_obs", "A number greater than or equal to 0 is expected. A non-numeric object was specified of type " + str(type(self.__train_df)))
        elif self.__nominal_new_category_min_obs < 0:
            error("nominal_new_category_min_obs", "A number greater than or equal to 0 is expected. A value of " + str(self.__mode_fallback_distinct_values) + "was specified.")

        #-----------------------------------------------------------------------
        # missing_indicators
        #-----------------------------------------------------------------------
        if type(self.__missing_indicators) != bool:
            error("missing_indicators", "Expecting a bool object with a value of True or False. Object of type " + str(type(self.__missing_indicators)) + " was specified.")

        #-----------------------------------------------------------------------
        # missing_indicators_min_obs
        #-----------------------------------------------------------------------
        if type(self.__missing_indicators_min_obs) not in [float, int]:
            error("missing_indicators_min_obs", "A number greater than or equal to 0 is expected. A non-numeric object was specified of type " + str(type(self.__missing_indicators_min_obs)))
        elif self.__missing_indicators_min_obs < 0:
            error("missing_indicators_min_obs", "A number greater than or equal to 0 is expected. A value of " + str(self.__missing_indicators_min_obs) + "was specified.")

        #-----------------------------------------------------------------------
        # model_x_columns
        #-----------------------------------------------------------------------
        if self.__method == 6 and type(self.__model_x_columns) == list and len(train_df_cols) > 0:
            train_missing_cols = [c for c in self.__model_x_columns if c not in train_df_cols]
            if len(train_missing_cols) > 0:
                for col in train_missing_cols:
                    error("model_x_columns", "Column " + col + " not found in the [train_df] dataframe.")

        if self.__method == 6 and type(self.__model_x_columns) == list and len(test_df_cols) > 0:
            test_missing_cols = [c for c in self.__model_x_columns if c not in test_df_cols]
            if len(test_missing_cols) > 0:
                for col in test_missing_cols:
                    error("model_x_columns", "Column " + col + " not found in the [test_df] dataframe.")

        return errors

    # -----------------------------------------------------------------------
    # Method: fit()
    # -----------------------------------------------------------------------
    def fit(self):

        _start_proc = time.time()
        _start_var = time.time()

        def proc_duration():
            nonlocal _start_proc
            dur = time.time() - _start_proc
            return dur

        def var_duration():
            nonlocal _start_var
            dur = time.time() - _start_var
            return dur

        errors = self.validate_properties()

        if len(errors) > 0:
            print(str(len(errors)) + " errors found with impute_missing.imputer properties:\n")
            for key in errors:
                print("   ERROR #" + str(key) + " - [" + str(errors[key][0]) + "] " + str(errors[key][1]))
            raise Exception(str(len(errors)) + " errors found with impute_missing.imputer properties.")

        def obs(x):
            if 0 < x < 1:
                rv = int(x * td_obs)
            else:
                rv = int(x)
            return rv

        #-----------------------------------------------------------------------
        # Label Column Name
        #-----------------------------------------------------------------------
        label_column = "" if self.__label is None else self.__label
        
        #-----------------------------------------------------------------------
        # List of Group By Column Names
        #-----------------------------------------------------------------------
        if self.__groupby is None or self.__groupby in ["", " ", [], [""], [" "]]:
            gbcolumns = ["z__groupby"]
        else:
            gbcolumns = self.__groupby

        #-----------------------------------------------------------------------
        # train_df # of Observations
        #-----------------------------------------------------------------------
        td_obs = len(self.__train_df)

        #-----------------------------------------------------------------------
        # List to capture the imputation rules for each column.
        #-----------------------------------------------------------------------
        rules_df = list()  #: type: pandas.DataFrame

        #-----------------------------------------------------------------------
        # Column count variable that will be incremented in the column loop
        # than follows.
        #-----------------------------------------------------------------------
        column_count = 0

        # -----------------------------------------------------------------------
        # List to capture missing imputations for the group by variables.
        # In practice this should already be done before using them as
        # in a groupby.  If not the mode() will be used.  This is not meant
        # to be optimal.  Handling of missing values in the groupby variables
        # should be handled manually with carefull inspection since they are
        # defining the population sub-segments that other analysis is being
        # done with each sub-segment.
        # -----------------------------------------------------------------------
        gb_impute_values = list()
        for gbcol in gbcolumns:
            mode = self.__train_df[gbcol].mode().tolist()[0]
            gb_impute_values.append([gbcol, mode])

        #-----------------------------------------------------------------------
        # Imputation Methods
        #-----------------------------------------------------------------------
        columns = [x for x in self.__columns if x not in gbcolumns + [label_column]]
        columns.sort()
        for col in columns:

            _start_var = time.time()

            col_dtype = self.__train_df[col].dtype
            if col_dtype in [numpy.dtype('float64'), numpy.dtype('float32'), numpy.dtype('int64'), numpy.dtype('int32')]:
                col_type = "Numeric"
            elif col_dtype == numpy.dtype('<m8[ns]'):
                col_type = "Interval"
            elif col_dtype == numpy.dtype('datetime64[ns]'):
                col_type = "Datetime"
            else:
                col_type = "Nominal"

            column_count += 1

            if col_type in ["Datetime", "Interval"]:
                if self.__verbose:
                    print("[%04d] %s - Excluded due to unsupported data type of %s (%s)."
                          % (column_count, col, col_type, col_dtype))
                    print(" ")
                continue

            if gbcolumns == ['__groupby__']:
                tdf = self.__train_df.loc[:, [x for x in [col, self.__label] if str(x) != ""]].copy()  # type: pandas.DataFrame
                tdf["z__groupby"] = 1
            else:
                tdf = self.__train_df.loc[:, [x for x in gbcolumns + [col, self.__label] if str(x) != ""]].copy()  # type: pandas.DataFrame

                #-----------------------------------------------------------------------
                # Group By variables can't have missing values and ideally should be
                # handled before using them here.  However, if missing values do exist
                # they will be assigned to the first mode of the non-missing values.
                #-----------------------------------------------------------------------
                if len(gb_impute_values) > 0:
                    for gbcol, gbval in gb_impute_values:
                        tdf[gbcol].fillna(gbval, inplace=True)

            df_unique = tdf[col].nunique()

            if df_unique == 0:
                if self.__verbose:
                    print("[%04d] %s - Excluded because all values are missing."
                          % (column_count, col))
                    print(" ")
                continue

            tdf["z__isnull"] = tdf[col].isnull().astype('float64')

            gb = tdf.groupby(gbcolumns, as_index=True)
            if len(gbcolumns) > 1:
                gbdf = pandas.DataFrame(index=pandas.MultiIndex.from_tuples(list(gb.indices.keys()), names=gbcolumns))  # type: pandas.DataFrame
            else:
                gbdf = pandas.DataFrame(index=pandas.Index(list(gb.indices.keys()), name=gbcolumns[0]))  # type: pandas.DataFrame

            gbdf["z__column_type"] = col_type

            gbdf["z__method"] = self.__method
            gbdf["z__nmiss"] = tdf.groupby(gbcolumns)["z__isnull"].sum()
            gbdf["z__nobs"] = tdf.groupby(gbcolumns)[col].count()

            gbdf["z__unique"] = tdf.groupby(gbcolumns)[col].nunique(dropna=True)

            gbdf["z__column"] = col

            gbdf["z__df_unique"] = df_unique
            gbdf["z__df_nmiss"] = tdf["z__isnull"].sum()
            gbdf["z__df_nobs"] = tdf[col].count()
            gbdf["z__df_mode"] = tdf[col].mode().tolist()[0]

            if col_type == "Numeric":
                gbdf["z__df_mean"] = tdf[col].mean()
                gbdf["z__mean"] = tdf.groupby(gbcolumns)[col].mean()

                gbdf["z__mode"] = tdf.groupby(gbcolumns)[col].\
                    apply(lambda x: pandas.Series.mode(x).tolist()).\
                    apply(lambda z: numpy.NaN if len(z) == 0 else z[0])

                gbdf["z__df_median"] = tdf[col].median()
                gbdf["z__median"] = tdf.groupby(gbcolumns)[col].median()
            else:
                gbdf["z__df_mean"] = None
                gbdf["z__mean"] = None
                gbdf["z__df_median"] = None
                gbdf["z__median"] = None
                gbdf["z__mode"] = tdf.groupby(gbcolumns)[col]. \
                    apply(lambda x: pandas.Series.mode(x).tolist()). \
                    apply(lambda z: numpy.NaN if len(z) == 0 else z[0])

            #-----------------------------------------------------------------------
            # Missing Indicators
            #-----------------------------------------------------------------------
            if not self.__missing_indicators:
                gbdf["z__indicator"] = False
            else:
                gbdf["z__indicator"] = (gbdf["z__nmiss_"] >= obs(self.__missing_indicators_min_obs))

            #-----------------------------------------------------------------------
            # Method in [1, 2] for Numeric Columns.
            #-----------------------------------------------------------------------
            if self.__method in [1, 2] and col_type == "Numeric":
                def calcn12(x: pandas.DataFrame, xmethod: int, xthreshhold: int, xtype: str = "impute"):
                    if x["z__unique"] == 1 and x["z__mode"] != 0:
                        return 0 if xtype == "impute" else "binary zero"
                    elif x["z__unique"] == 1 and x["z__mode"] != 1:
                        return 1 if xtype == "impute" else "binary one"
                    elif x["z__nobs"] > 0 and x["z__unique"] > 0:
                        if x["z__unique"] <= xthreshhold:
                            return x["z__mode"] if xtype == "impute" else "mode"
                        elif xmethod == 1:
                            return x["z__mean"] if xtype == "impute" else "mean"
                        elif xmethod == 2:
                            return x["z__median"] if xtype == "impute" else "median"
                    else:
                        if x["z__df_unique"] <= xthreshhold:
                            return x["z__df_mode"] if xtype == "impute" else "mode"
                        elif xmethod == 1:
                            return x["z__df_mean"] if xtype == "impute" else "mean"
                        elif xmethod == 2:
                            return x["z__df_median"] if xtype == "impute" else "median"

                gbdf["z__impute"] = gbdf.apply(lambda z: calcn12(z, self.__method, self.__mode_fallback_distinct_values, "impute"), axis=1).astype('float64')
                gbdf["z__impute_type"] = gbdf.apply(lambda z: calcn12(z, self.__method, self.__mode_fallback_distinct_values, "impute_type"), axis=1).astype('str')
                rules_df.append([column_count, col, self.__method, gbdf.copy(deep=True)])

            #-----------------------------------------------------------------------
            # Method in [1, 2] for Nonminal Columns
            #-----------------------------------------------------------------------
            elif self.__method in [1, 2] and col_type == "Nominal":
                def calcn(x: pandas.DataFrame, xtype: str = "impute"):
                    if x["z__unique"] == 1 and x["z__mode"] != "[nan]":
                        return "[nan]" if xtype == "impute" else "new category"
                    elif x["z__unique"] == 1 and x["z__mode"] != "[na]":
                        return "[na]" if xtype == "impute" else "new category"
                    elif x["z__nobs"] > 0 and x["z__unique"] > 0:
                        return x["z__mode"] if xtype == "impute" else "mode"
                    else:
                        return x["z__df_mode"] if xtype == "impute" else "mode"

                gbdf["z__impute"] = gbdf.apply(lambda z: calcn(z, "impute"), axis=1).astype('str')
                gbdf["z__impute_type"] = gbdf.apply(lambda z: calcn(z, "impute__type"), axis=1).astype('str')
                rules_df.append([column_count, col, self.__method, gbdf.copy(deep=True)])

            #-----------------------------------------------------------------------
            # Method = 3 for Numeric, Nominal, and DateTime Columns
            #-----------------------------------------------------------------------
            elif self.__method == 3:
                def calc3(x: pandas.DataFrame, xcoltype: str, xtype: str = "impute"):
                    if xcoltype == "Numeric":
                        if x["z__unique"] == 1 and x["z__mode"] != 0:
                            return 0 if xtype == "impute" else "binary zero"
                        elif x["z__unique"] == 1 and x["z__mode"] != 1:
                            return 1 if xtype == "impute" else "binary one"
                        elif x["z__nobs"] > 0 and x["z__unique"] > 0:
                            return x["z__mode"] if xtype == "impute" else "mode"
                        else:
                            return x["z__df_mode"] if xtype == "impute" else "mode"
                    else:
                        if x["z__unique"] == 1 and x["z__mode"] != "[nan]":
                            return "[nan]" if xtype == "impute" else "new category"
                        elif x["z__unique"] == 1 and x["z__mode"] != "[na]":
                            return "[na]" if xtype == "impute" else "new category"
                        elif x["z__nobs"] > 0 and x["z__unique"] > 0:
                            return x["z__mode"] if xtype == "impute" else "mode"
                        else:
                            return x["z__df_mode"] if xtype == "impute" else "mode"

                gbdf["z__impute"] = gbdf.apply(lambda z: calc3(z, col_type, "impute"), axis=1)
                gbdf["z__impute_type"] = gbdf.apply(lambda z: calc3(z, col_type, "impute_type"), axis=1)
                rules_df.append([column_count, col, self.__method, gbdf.copy(deep=True)])

            #-----------------------------------------------------------------------
            # Method in [4, 5] for Numeric Columns
            #-----------------------------------------------------------------------
            elif self.__method in [4, 5] and col_type == "Numeric":
                tdf.eval('z__miss_x_resp = z__isnull * ' + label_column, inplace=True)
                tdf["z__bins"] = tdf.groupby(gbcolumns)[col]. \
                    transform(lambda x: pandas.qcut(x, self.__mean_label_bins, labels=False, duplicates="drop")).astype('float64')
                tdf["z__bins"].fillna(99999.0, inplace=True)

                gbdf["z__missing_sumresp"] = tdf.groupby(gbcolumns)['z__miss_x_resp'].sum().astype('float64')
                gbdf["z__missing_avgresp"] = gbdf.apply(lambda x: x['z__missing_sumresp'] / x['z__nmiss'] if x['z__nmiss'] > 0 else numpy.NaN, axis=1).astype('float64')

                # _html = gbdf.to_html(justify="right", notebook=True, max_rows=20)
                # display(HTML(_html))

                q_cols_a = gbcolumns[0]
                q_join_criteria = "a.%s = b.%s" % (gbcolumns[0], gbcolumns[0])
                if len(gbcolumns) > 1:
                    for i in range(1, len(gbcolumns)):
                        q_cols_a += ", " + gbcolumns[i]
                        q_join_criteria += " and a.%s = b.%s" % (gbcolumns[i], gbcolumns[i])
                q_cols_b = q_cols_a
                q_cols_a += ", z__bins"

                if 0 <= self.__mean_label_min_obs <= 1:
                    q_min_label_min_obs = "cast(max(" + str(self.__mean_label_min_obs) + "*b.z__nobs,2) as integer)"
                else:
                    q_min_label_min_obs = "cast(max(" + str(self.__mean_label_min_obs) + ",2) as integer)"

                q = f'''
                    select a.*, 
                           {q_min_label_min_obs} as z__min_label_min_obs
                    from (select {q_cols_a}
                                 ,avg({label_column}) as z__bin_avgresp
                                 ,count(*) as z__bin_nobs
                                 ,avg({col}) as z__bin_col_mean
                          from tdf
                          where {col} is not NULL 
                          group by {q_cols_a}
                          ) a 
                    join (select {q_cols_b}, z__nobs
                          from gbdf                                              
                          ) b
                       on {q_join_criteria}
                    order by {q_cols_a}
                    '''
                # print(q)
                bin_gbdf = pandasql.sqldf(q)

                bin_gbdf["z__bin_col_median"] = tdf.query(col + " == " + col).groupby(gbcolumns + ["z__bins"], as_index=False)[col].median()[col]
                bin_gbdf.reset_index(inplace=True)
                gbdf.reset_index(inplace=True)
                bin_gbdf = pandas.merge(gbdf, bin_gbdf, on=gbcolumns, how='outer')  # type: pandas.DataFrame
                bin_gbdf.eval("z__diff = abs(z__bin_avgresp - z__missing_avgresp)", inplace=True)

                def adj_diff(x):
                    if x["z__bin_nobs"] >= x["z__min_label_min_obs"] and x["z__nmiss"] >= x["z__min_label_min_obs"]:
                        return x["z__diff"]
                    else:
                        return numpy.NaN

                bin_gbdf["z__adj_diff"] = bin_gbdf.apply(lambda z: adj_diff(z), axis=1)
                bin_gbdf["z__adj_diff"] = bin_gbdf.apply(lambda z: numpy.NaN if z["z__bins"] == 99999.0 else z["z__adj_diff"], axis=1)
                bin_gbdf["z__adj_diff"].fillna(value=999999999999.0, inplace=True)

                # bin_gbdf["z__adj_diff_rank"] = bin_gbdf.groupby(gbcolumns)["z__adj_diff", "z_bins"].rank(method='dense')

                bin_gbdf.sort_values(by=gbcolumns + ["z__adj_diff", "z__bins"], inplace=True, axis=0)

                # bin_gbdf.query("z__adj_diff_rank == 1", inplace=True)
                bin_gbdf = bin_gbdf.groupby(gbcolumns, as_index=False).first()
                bin_gbdf.set_index(gbcolumns, drop=True, append=False, inplace=True)

                # _html = bin_gbdf.to_html(justify="right", notebook=True)
                # display(HTML(_html))

                def calcn45(x: pandas.DataFrame, xmethod: int, xthreshhold: int, xtype: str = "impute"):
                    if x["z__unique"] == 1 and x["z__mode"] != 0:
                        return 0 if xtype == "impute" else "binary zero"
                    elif x["z__unique"] == 1 and x["z__mode"] != 1:
                        return 1 if xtype == "impute" else "binary one"
                    elif x["z__adj_diff"] < 999999999999.0:
                        return x["z__bin_col_mean"] if xtype == "impute" else "mean response"
                    elif x["z__nobs"] > 0 and x["z__unique"] > 0:
                        if x["z__unique"] <= xthreshhold:
                            return x["z__mode"] if xtype == "impute" else "mode"
                        elif xmethod == 4:
                            return x["z__mean"] if xtype == "impute" else "mean"
                        elif xmethod == 5:
                            return x["z__median"] if xtype == "impute" else "median"
                    else:
                        if x["z__df_unique"] <= xthreshhold:
                            return x["z__df_mode"] if xtype == "impute" else "mode"
                        elif xmethod == 4:
                            return x["z__df_mean"] if xtype == "impute" else "mean"
                        elif xmethod == 5:
                            return x["z__df_median"] if xtype == "impute" else "median"

                bin_gbdf["z__impute"] = bin_gbdf.apply(lambda z: calcn45(z, self.__method, self.__mode_fallback_distinct_values, "impute"), axis=1)
                bin_gbdf["z__impute_type"] = bin_gbdf.apply(lambda z: calcn45(z, self.__method, self.__mode_fallback_distinct_values, "impute_type"), axis=1)

                rules_df.append([column_count, col, self.__method, bin_gbdf.copy(deep=True)])

            #-----------------------------------------------------------------------
            # Method in [4, 5] for Nominal Columns
            #-----------------------------------------------------------------------
            elif self.__method in [4, 5] and col_type == "Nominal":
                tdf.eval('z__miss_x_resp = z__isnull * ' + label_column, inplace=True)

                gbdf["z__missing_sumresp"] = tdf.groupby(gbcolumns)['z__miss_x_resp'].sum().astype('float64')
                gbdf["z__missing_avgresp"] = gbdf.apply(lambda x: x['z__missing_sumresp'] / x['z__nmiss'] if x['z__nmiss'] > 0 else numpy.NaN, axis=1).astype('float64')

                # _html = gbdf.to_html(justify="right", notebook=True, max_rows=20)
                # display(HTML(_html))

                q_cols_a = gbcolumns[0]
                q_join_criteria = "a.%s = b.%s" % (gbcolumns[0], gbcolumns[0])
                if len(gbcolumns) > 1:
                    for i in range(1, len(gbcolumns)):
                        q_cols_a += ", " + gbcolumns[i]
                        q_join_criteria += " and a.%s = b.%s" % (gbcolumns[i], gbcolumns[i])
                q_cols_b = q_cols_a
                q_cols_a += ", " + col

                if 0 <= self.__mean_label_min_obs <= 1:
                    q_min_label_min_obs = "cast(max(" + str(self.__mean_label_min_obs) + "*b.z__nobs,2) as integer)"
                else:
                    q_min_label_min_obs = "cast(max(" + str(self.__mean_label_min_obs) + ",2) as integer)"

                q = f'''
                    select a.*, 
                           {q_min_label_min_obs} as z__min_label_min_obs
                    from (select {q_cols_a}
                                 ,avg({label_column}) as z__bin_avgresp
                                 ,count(*) as z__bin_nobs
                          from tdf
                          where {col} is not NULL 
                          group by {q_cols_a}
                          ) a 
                    join (select {q_cols_b}, z__nobs
                          from gbdf                                              
                          ) b
                       on {q_join_criteria}
                    order by {q_cols_a}
                    '''
                # print(q)
                bin_gbdf = pandasql.sqldf(q)

                bin_gbdf.reset_index(inplace=True)
                gbdf.reset_index(inplace=True)

                bin_gbdf = pandas.merge(gbdf, bin_gbdf, on=gbcolumns, how='outer')  # type: pandas.DataFrame
                bin_gbdf.eval("z__diff = abs(z__bin_avgresp - z__missing_avgresp)", inplace=True)

                def adj_diff(x):
                    if x["z__bin_nobs"] >= x["z__min_label_min_obs"] and x["z__nmiss"] >= x["z__min_label_min_obs"]:
                        return x["z__diff"]
                    else:
                        return numpy.NaN

                bin_gbdf["z__adj_diff"] = bin_gbdf.apply(lambda z: adj_diff(z), axis=1)
                # bin_gbdf["z__adj_diff"] = bin_gbdf.apply(lambda z: numpy.NaN if z["z__bins"] == 99999.0 else z["z__adj_diff"], axis=1)
                bin_gbdf["z__adj_diff"].fillna(value=999999999999.0, inplace=True)

                bin_gbdf.sort_values(by=gbcolumns + ["z__adj_diff", col], inplace=True, axis=0)
                bin_gbdf = bin_gbdf.groupby(gbcolumns, as_index=False).first()

                bin_gbdf.set_index(gbcolumns, drop=True, append=False, inplace=True)

                def calcn45(x: pandas.DataFrame, xtype: str = "impute"):
                    if x["z__unique"] == 1 and x["z__mode"] != "[nan]":
                        return "[nan]" if xtype == "impute" else "new category"
                    elif x["z__unique"] == 1 and x["z__mode"] != "[na]":
                        return "[na]" if xtype == "impute" else "new category"
                    elif x["z__adj_diff"] < 999999999999.0:
                        return x[col] if xtype == "impute" else "mean response"
                    else:
                        return x["z__mode"] if xtype == "impute" else "mode"

                bin_gbdf["z__impute"] = bin_gbdf.apply(lambda z: calcn45(z, "impute"), axis=1)
                bin_gbdf["z__impute_type"] = bin_gbdf.apply(lambda z: calcn45(z, "impute_type"), axis=1)

                rules_df.append([column_count, col, self.__method, bin_gbdf.copy(deep=True)])

            #-----------------------------------------------------------------------
            # [End of Loop] - for col in [x for x in self.__columns if x not in gbcolumns]:
            #-----------------------------------------------------------------------

            if self.__verbose:
                print("[%04d] %s - Processed in %s seconds." % (column_count, col, var_duration()))
                if col_type == "Numeric":
                    print("       dtype=%s(%s)  NaN=%s  Unique=%s  Mean=%s  Median=%s"
                          % (col_type, col_dtype,
                             gbdf["z__df_nmiss"].min(),
                             gbdf["z__df_unique"].min(),
                             gbdf["z__df_mean"].min(),
                             gbdf["z__df_median"].min()
                             )
                          )
                else:
                    print("       dtype=%s(%s)  NaN=%s  Unique=%s  Mode=%s"
                          % (col_type, col_dtype,
                             gbdf["z__df_nmiss"].min(),
                             gbdf["z__df_unique"].min(),
                             gbdf["z__df_mode"].min()
                             )
                          )
                print(" ")
        #-----------------------------------------------------------------------
        # Return Rules Class Object
        #-----------------------------------------------------------------------
        if self.__verbose:
            print(" ")
            print("Imputer.fit() finished in %s seconds." % proc_duration())
            print("%s variables processed." % len([x for x in self.__columns if x not in gbcolumns + [label_column]]))

        if len(rules_df) > 0:
            return ImputeRules(imputed_columns=[x for x in self.__columns if x not in gbcolumns + [label_column]],
                               label=label_column,
                               groupby=gbcolumns if gbcolumns != ["z__groupby"] else list(),
                               groupby_impute_values=gb_impute_values if gbcolumns != ["z__groupby"] else list(),
                               create_dt=datetime.datetime.now(),
                               notes=None,
                               rules_list=rules_df
                               )
        else:
            raise Exception("ERROR:  No columns found to process.  Datetime and Interval columns are not supported and skipped!")
