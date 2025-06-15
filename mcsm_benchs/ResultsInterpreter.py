"""mcms-benchs: A toolbox that provides a common framework for benchmarks of multi-component signal processing methods.

Copyright (C) 2024  Juan Manuel Miramont

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import seaborn as sns
from mcsm_benchs.Benchmark import Benchmark
import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
import string
import os
import plotly.express as px
import scipy.stats as spst


class ResultsInterpreter:
    """This class takes a Benchmark-class object to produce a series of plots and tables
    summarizing the obtained results:

    Methods
    -------
    def get_benchmark_as_data_frame(self):

    """

    def __init__(self, a_benchmark):
        if not isinstance(a_benchmark, Benchmark):
            # self.results = a_benchmark.get_results_as_df()
            # else:
            raise ValueError("Input should be a Benchmark.\n")

        self.benchmark = a_benchmark
        self.task = a_benchmark.task
        self.methods_ids = a_benchmark.methods_ids
        self.N = a_benchmark.N
        self.repetitions = a_benchmark.repetitions
        self.snr_values = a_benchmark.SNRin
        self.signal_ids = a_benchmark.signal_ids
        self.methods_and_params_dic = a_benchmark.methods_and_params_dic

        # Number of elements to display in the plots.
        self.ncomparisons = sum(
            [len(self.methods_and_params_dic[m]) for m in self.methods_ids]
        )

        self.path_results = os.path.join("results")
        self.path_results_figures = os.path.join("results", "figures")

    # --------------------------------------------------------------------------------------

    def get_benchmark_as_data_frame(self, idx=0):
        """Returns a DataFrame with the raw data produced by the benchmark with the
        following format:
            -------------------------------------------------------------------------
            | Method | Parameter | Signal_id | Repetition | SNRin_1 | ... | SNRin_n |
            -------------------------------------------------------------------------

        Returns:
            DataFrame: Raw data of the comparisons.
        """
        df = self.benchmark.get_results_as_df()
        if type(df) == list:
            df = df[idx]

        return df

    # --------------------------------------------------------------------------------------

    def rearrange_data_frame(self, results=None, idx=0):
        """Rearrange DataFrame table for using Seaborn library.

        Args:
            results (DataFrame, optional): If not None, must receive the DataFrame
            produced by a Benchmark-class object using get_results_as_df(). If None,
            uses the Benchmark object given to the constructor of the Interpreter.
            Defaults to None.

        Returns:
            DataFrame: Rearranged DataFrame
        """
        if results is None:
            df = self.benchmark.get_results_as_df()
            if type(df) == list:
                df = df[idx]
        else:
            df = results

        aux_dic = dict()
        new_columns = df.columns.values[0:5].copy()

        new_columns[-1] = "QRF"

        for i in range(4, df.shape[1]):
            idx = [j for j in range(4)] + [i]
            df_aux = df.iloc[:, idx]
            df_aux.columns = new_columns
            aux_dic[df.columns[i]] = df_aux

        df3 = pd.concat(aux_dic, axis=0)
        df3 = df3.reset_index()
        df3 = df3.drop(labels="level_1", axis=1)
        df3.columns.values[0] = "SNRin"
        return df3

    # --------------------------------------------------------------------------------------

    def _get_df_means(self, df=None, idx=0):
        """Generates a DataFrame of mean results to .md file.

        Returns:
            str: String containing the table.
        """
        # if filename is None:
        # filename = 'results'
        if df is None:
            df = self.benchmark.get_results_as_df()
            if type(df) == list:
                df = df[idx]

        # Check if matrix values are bools:
        aux = df.iloc[:, 4::].to_numpy()
        if type(aux[0, 0]) is bool:
            df.iloc[:, 4::] = aux.astype(int)

        column_names = ["Method + Param"] + [col for col in df.columns.values[4::]]
        df_means = list()

        for signal_id in self.signal_ids:
            methods_names = list()
            snr_out_mean = np.zeros((1, len([col for col in df.columns.values[4::]])))
            snr_out_std = np.zeros((1, len([col for col in df.columns.values[4::]])))
            aux_dic = dict()

            df2 = df[df["Signal_id"] == signal_id]
            for metodo in self.methods_and_params_dic:
                tag = metodo
                aux = df2[df2["Method"] == metodo]
                if len(self.methods_and_params_dic[metodo]) > 1:
                    for params in self.methods_and_params_dic[metodo]:
                        methods_names.append(tag + params)
                        valores = df2[
                            (df2["Method"] == metodo) & (df2["Parameter"] == params)
                        ]
                        # Compute the means
                        valores_mean = valores.iloc[:, 4::].to_numpy().mean(axis=0)
                        valores_mean.resize((1, valores_mean.shape[0]))
                        snr_out_mean = np.concatenate((snr_out_mean, valores_mean))
                        # Compute the std
                        # valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                        # valores_std.resize((1,valores_std.shape[0]))
                        # snr_out_std = np.concatenate((snr_out_std,valores_std))

                else:
                    methods_names.append(tag)
                    valores = df2[df2["Method"] == metodo]
                    # Compute the means
                    valores_mean = valores.iloc[:, 4::].to_numpy().mean(axis=0)
                    valores_mean.resize((1, valores_mean.shape[0]))
                    snr_out_mean = np.concatenate((snr_out_mean, valores_mean))
                    # Compute the std
                    # valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                    # valores_std.resize((1,valores_std.shape[0]))
                    # snr_out_std = np.concatenate((snr_out_std,valores_std))

            snr_out_mean = snr_out_mean[1::]
            # snr_out_std = snr_out_std[1::]
            aux_dic[column_names[0]] = methods_names
            for i in range(1, len(column_names)):
                aux_dic[str(column_names[i])] = snr_out_mean[:, i - 1]

            df_means.append(pd.DataFrame(aux_dic))

        return df_means

    # ---------------------------------------------------------------------------------------

    def _get_df_std(self, df=None, varfun=None, idx=0):
        """Generates a DataFrame with a variability function from the results.
        Returns an string to create a .md file.

        Args:
            df (_type_, optional): Main DataFrame with the results. Defaults to None.
            varfun (str, optional): A variability function. Defaults to 'std', which uses numpy.std().
            idx (int, optional): Index of the corresponding performance function. Defaults to 0.

        Returns:
            Two DataFrames with the lower and upper variability indicator.
        """
        # Use the benchmark results if no DataFrame is passed as parameter.
        if df is None:
            df = self.benchmark.get_results_as_df()
            if type(df) == list:
                df = df[idx]

        # Choose an appropriate variability measure:
        if varfun is None:
            if self.task == "detection":  # Bonferroni corrected Clopper-Pearson CI
                varfun = lambda x: _clopper_pearson(
                    x,
                    alpha=0.05,
                    bonferroni=self.ncomparisons,
                )
            # For denoising or misc, use bootstrap CI.
            else:
                # varfun = lambda x: (np.std(x),np.std(x))
                varfun = lambda x: _ci(
                    x,
                    alpha=0.05,
                    bonferroni=self.ncomparisons,
                )

        # Check if matrix values are bools:
        aux = df.iloc[:, 4::].to_numpy()
        if type(aux[0, 0]) is bool:
            df.iloc[:, 4::] = aux.astype(int)

        column_names = ["Method + Param"] + [col for col in df.columns.values[4::]]
        output_string = ""
        df_std_upper = list()
        df_std_lower = list()

        for signal_id in self.signal_ids:
            methods_names = list()
            snr_out_std = np.zeros(
                (1, 2 * len([col for col in df.columns.values[4::]]))
            )
            aux_dic = dict()
            aux_dic_2 = dict()

            df2 = df[df["Signal_id"] == signal_id]
            for metodo in self.methods_and_params_dic:
                tag = metodo
                aux = df2[df2["Method"] == metodo]
                if len(self.methods_and_params_dic[metodo]) > 1:
                    for params in self.methods_and_params_dic[metodo]:
                        methods_names.append(tag + params)
                        valores = df2[
                            (df2["Method"] == metodo) & (df2["Parameter"] == params)
                        ]
                        # Compute the std
                        valores_std = np.array(
                            [varfun(col) for col in valores.iloc[:, 4::].to_numpy().T]
                        )
                        valores_std.resize((1, valores_std.size))
                        snr_out_std = np.concatenate((snr_out_std, valores_std))

                else:
                    methods_names.append(tag)
                    valores = df2[df2["Method"] == metodo]
                    # Compute the std
                    # valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                    valores_std = np.array(
                        [varfun(col) for col in valores.iloc[:, 4::].to_numpy().T]
                    )
                    valores_std.resize((1, valores_std.size))
                    snr_out_std = np.concatenate((snr_out_std, valores_std))

            snr_out_std = snr_out_std[1::]
            aux_dic[column_names[0]] = methods_names
            for i, j in zip(
                range(1, len(column_names)), range(0, snr_out_std.shape[1], 2)
            ):
                aux_dic[str(column_names[i])] = snr_out_std[:, j]

            df_std_lower.append(pd.DataFrame(aux_dic))

            aux_dic_2[column_names[0]] = methods_names
            for i, j in zip(
                range(1, len(column_names)), range(1, snr_out_std.shape[1], 2)
            ):
                aux_dic_2[str(column_names[i])] = snr_out_std[:, j]

            df_std_upper.append(pd.DataFrame(aux_dic_2))

        return df_std_upper, df_std_lower

    # --------------------------------------------------------------------------------------
    def _get_table_means_and_std(self, link="", pm_name=None, idx=0):
        """Generates a table of mean and std results to .md file.
        Highlights the best results.

        Returns:
            str: String containing the table.
        """
        task = self.benchmark.task
        df = self.benchmark.get_results_as_df()
        if type(df) == list:
            df = df[idx]

        column_names = ["Method + Param"] + [col for col in df.columns.values[4::]]
        output_string = ""

        # Get dataframes with means and some variability measure
        dfs_means = self._get_df_means()
        dfs_std_upper, dfs_std_lower = self._get_df_std()

        for dfu, dfl, dfm in zip(dfs_std_upper, dfs_std_lower, dfs_means):
            dfu.iloc[:, 1::] = dfu.iloc[:, 1::] + dfm.iloc[:, 1::]
            dfl.iloc[:, 1::] = dfm.iloc[:, 1::] - dfl.iloc[:, 1::]

        # TODO: Round std to 2 decimal places.
        # def round_to_2dec(input):
        # if type(input) is float:
        # return '{:.2f}'.format(input)

        # dfs_std_lower = [df.applymap(round_to_2dec) for df in dfs_std_lower]
        # dfs_std_upper = [df.applymap(round_to_2dec) for df in dfs_std_upper]

        dfs_std = [
            {
                col: [[low, upp] for low, upp in zip(dfl[col], dfu[col])]
                for col in dfu.columns.values[1::]
            }
            for dfu, dfl in zip(dfs_std_upper, dfs_std_lower)
        ]
        dfs_std = [pd.DataFrame(df) for df in dfs_std]

        for signal_id, df_means, df_std in zip(self.signal_ids, dfs_means, dfs_std):
            df_means_aux = df_means.copy()

            # Format all floats with two decimals:
            nparray_aux = df_means.iloc[:, 1::].to_numpy()
            maxinds = np.argmax(nparray_aux, axis=0)

            for col, max_ind in enumerate(maxinds):
                for i in range(
                    df_means_aux.shape[0]
                ):  # range(len(df_means.loc[col+1])):
                    df_means_aux.iloc[i, col + 1] = "{:.2f}".format(
                        df_means.iloc[i, col + 1]
                    )

                #  Also, highlight maxima
                df_means_aux.iloc[max_ind, col + 1] = (
                    "**" + "{:.2f}".format(df_means.iloc[max_ind, col + 1]) + "**"
                )

            # Change column names to make it more human-readable
            df_results = pd.DataFrame()
            df_results[column_names[0]] = df_means[column_names[0]]
            for col_ind in range(1, len(column_names)):
                df_results["SNRin=" + str(column_names[col_ind]) + "dB (average)"] = (
                    df_means_aux[str(column_names[col_ind])]
                )
                # df_results['SNRin='+str(column_names[col_ind])+'dB (CI)'] = df_std[str(column_names[col_ind])]

                # Make sure to have two decimals precision in each std column.
                df_results["SNRin=" + str(column_names[col_ind]) + "dB (CI)"] = [
                    ["{:.2f}".format(a) for a in row]
                    for row in df_std[str(column_names[col_ind])]
                ]

                # df_results['SNRin='+str(column_names[col_ind])+'dB (CI)'] = df_std[str(column_names[col_ind])]

            # Make sure having two decimals precision in each std column

            if pm_name is not None:
                df_results.rename(columns={"QRF": pm_name})

            # Table header with links
            aux_string = (
                "### Signal: "
                + signal_id
                + " [[View Plot]]("
                + link
                + "/plot_"
                + signal_id
                + ".html)  "
                + "  [[Get .csv]]("
                + link
                + "/results_"
                + signal_id
                + ".csv"
                + ")"
                + "\n"
                + df_results.to_markdown(floatfmt=".2f")
                + "\n"
            )
            output_string += aux_string

        return output_string

    # --------------------------------------------------------------------------------------
    def _get_report_preamble(self, link=None):
        """Creates the preamble of the .md file with a table summarizing the benchmark results.

        Returns:
            str: String with the table header.
        """

        if link is None:
            link_text = ""
        else:
            link_text = " [[Results .csv]](" + link + "/results.csv" + ") " + "\n"

        lines = [
            "# Benchmark Report" + link_text + "\n",
            "## Configuration" + "\n",
            # 'Parallelize' + str(self.benchmark.parallel_flag) + '\n',
            "Length of signals: " + str(self.N) + "\n",
            "Repetitions: " + str(self.repetitions) + "\n",
            "SNRin values: ",
        ]

        lines = lines + [str(val) + ", " for val in self.snr_values] + ["\n"]
        lines = (
            lines
            + ["### Methods  \n"]
            + ["* " + methid + " \n" for methid in self.methods_ids]
        )
        lines = (
            lines
            + ["### Signals  \n"]
            + ["* " + signid + " \n" for signid in self.signal_ids]
        )
        # lines = lines + ['## Figures:\n ![Summary of results](results/../figures/plots_grid.png) \n']
        lines = lines + ["## Mean results tables: \n"]

        # TODO Check this part for different perf. functions
        if self.task == "denoising":
            lines = lines + [
                "The results shown here are the average and 95\% CI of \
                              the performance metric with Bonferroni correction. \
                              Best performances are **bolded**. \n"
            ]

        if self.task == "detection":
            lines = lines + [
                "The results shown here are the average and 95\% Clopper-Pearson CI of \
                            the estimated detection power with Bonferroni correction. \
                            Best performances are **bolded**. \n"
            ]

        return lines

    # --------------------------------------------------------------------------------------
    def save_report(
        self, filename=None, path="results", bars=False, link="", pm_name=None
    ):
        """This function generates a report of the results given in the Benchmark-class
        object. The report is saved in a Markdown syntax to be viewed as a .md file,
        while a .csv file is generated with the results.

        Args:
            filename (str, optional): Path for saving the report. Defaults to None.
        """
        if filename is None:
            filename = "results_" + self.task + ".md"

        # Check path availability, otherwise create folder.
        if not os.path.exists(path):
            os.makedirs(path)

        output_path = os.path.join(path, filename)

        # Generate table header:
        lines = self._get_report_preamble(link=link)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
            # f.writelines(lines)

        # Append table under header
        table_string = self._get_table_means_and_std(link=link, pm_name=pm_name)
        with open(output_path, "a") as f:
            f.write(table_string)

        return True

    # -------------------------------------------------------------------------------------
    def get_snr_plot(
        self,
        df,
        x=None,
        y=None,
        hue=None,
        axis=None,
        ylabel=None,
        plot_type="linear",
        **kwargs
    ):
        """Generates a Quality Reconstruction Factor (QRF) vs. SNRin plot. The QRF is
        computed as:
        QRF = 20 log ( norm(x) / norm(x-x_r)) [dB]
        where x is the noiseless signal and x_r is de denoised estimation of x.

        Args:
            df (DataFrame): DataFrame with the results of the simulation.
            x (str, optional): Column name to use as the horizontal axis.
            Defaults to None.
            y (str, optional): Column name to use as the vertical axis.
            Defaults to None.
            hue (str, optional): Column name with the methods' name. Defaults to None.
            axis (matplotlib.Axes, optional): The axis object where the plot will be
            generated. Defaults to None.
        """

        markers = ["o", "d", "s", "*"]
        aux = np.unique(df[hue].to_numpy())
        # print(aux)
        # fig, axis2 = plt.subplots(1,1)

        plots = [
            (method_name, markers[np.mod(i, 4)]) for i, method_name in enumerate(aux)
        ]
        u_offset = np.linspace(-2, 2, len(plots))
        u_offset = np.zeros_like(u_offset)
        for offs_idx, plots_info in enumerate(plots):
            method_name, marker = plots_info
            df_aux = df[df[hue] == method_name]
            u = np.unique(df_aux[x].to_numpy())
            v = np.zeros_like(u, dtype=float)

            label = "".join(
                [c for c in string.capwords(method_name, sep="_") if c.isupper()]
            )
            if method_name.find("-") > -1:
                label = label + method_name[method_name.find("-") : :]

            for uind, j in enumerate(u):
                df_aux2 = df_aux[df_aux[x] == j]
                no_nans = df_aux2[y].dropna()
                if no_nans.shape != (0,):
                    v[uind] = np.nanmean(no_nans.to_numpy())

            if plot_type == "linear":
                axis.plot(
                    u + u_offset[offs_idx],
                    v,
                    "-" + marker,
                    # ms = 5,
                    # linewidth = 1.0,
                    label=label,
                    **kwargs
                )

            if plot_type == "semilogy":
                axis.semilogy(
                    u + u_offset[offs_idx],
                    v + 1e-10,
                    "-" + marker,
                    label=label,
                    **kwargs
                )
                axis.yaxis.set_minor_locator(
                    mticker.LogLocator(numticks=999, subs="auto")
                )

        # axis.plot([np.min(u), np.max(u)],[np.min(u), np.max(u)],'r',
        # linestyle = (0, (5, 10)),
        # linewidth = 0.75)
        axis.set_xticks(u)
        axis.set_yticks(v)
        # axis.set_xlabel(x + ' (dB)')
        # axis.set_ylabel(y + ' (dB)')

        axis.set_xlabel("SNRin (dB)")

        if ylabel is None:
            if self.benchmark.task == "denoising":
                axis.set_ylabel("QRF (dB)")

            if self.benchmark.task == "detection":
                axis.set_ylabel("Detection Power")
        else:
            axis.set_ylabel(ylabel)

        return True

    # --------------------------------------------------------------------------------------
    def get_snr_plot_bars(
        self,
        df,
        x=None,
        y=None,
        hue=None,
        errbar_fun=("ci", 95),
        axis=None,
        errbar_params=None,
    ):
        """Generates a Quality Reconstruction Factor (QRF) vs. SNRin barplot.
        The QRF is computed as:
                        QRF = 20 log ( norm(x) / norm(x-x_r)) [dB]
        where x is the noiseless signal and x_r is de denoised estimation of x.

        Args:
            df (DataFrame): DataFrame with the results of the simulation.
            x (str, optional): Column name to use as the horizontal axis.
            Defaults to None.
            y (str, optional): Column name to use as the vertical axis.
            Defaults to None.
            hue (str, optional): Column name with the methods' name. Defaults to None.
            axis (matplotlib.Axes, optional): The axis object where the plot will be
            generated. Defaults to None.
        """

        if errbar_params is None:
            errbar_params = {
                "errwidth": 0.1,
                "capsize": 0.02,
            }

        barfig = sns.barplot(
            x="SNRin",
            y="QRF",  # It's QRF in the DataFrame, but changed later.
            hue="Method",
            data=df,
            dodge=True,
            errorbar=errbar_fun,
            ax=axis,
            **errbar_params
        )

        axis.set_xlabel("SNRin (dB)")
        if self.benchmark.task == "denoising":
            axis.set_ylabel("QRF (dB)")

        if self.benchmark.task == "detection":
            axis.set_ylabel("Detection Power")

        return barfig

    # --------------------------------------------------------------------------------------
    def get_summary_plots(
        self,
        df_rearr=None,
        size=(3, 3),
        savetofile=True,
        path=None,
        filter_crit="all",
        filter_str=None,
        errbar_fun=("ci", 95),
        errbar_params=None,
        ax=None,
        plot_type="bars",
        bars=True,
        magnitude="absolute",
    ):
        """Generates individual performance plots for each signal, displaying the
        performance measure of all methods for all noise conditions.

        Args:
            size (tuple, optional): Size (in inches) of the figures. Defaults to (3,3).
            savetofile (bool, optional): Whether to save or not the figures.
            Defaults to True.
            filename (_type_, optional): Path and file name to save the figures. If None
            figures are saved in "results/figures" . Defaults to None.
            filter_str (_type_, optional): A string, or a list of strings, to select
            the methods to plot. If None, plots all the methods. Defaults to None.
            filter_crit (str, optional): A criterion to use the strings passed in
            filter_str. If 'all', only choose those methods where all the strings
            appear in the "Method" column of the results DataFrame. If 'any', select the methods for which any of the strings appear in the mentioned column.
            Defaults to 'all'.

            plot_type (str, optional): _description_. Defaults to 'lines'.

        Returns:
            list: A list with matplotlib figures.
        """

        # Nsignals = len(self.signal_ids)

        if df_rearr is None:
            df_rearr = self.rearrange_data_frame()

        if magnitude == "difference":
            df_rearr["QRF"] = df_rearr["QRF"] - df_rearr["SNRin"]

        list_figs = list()

        for signal_id in self.signal_ids:
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            else:
                fig = plt.gca()

            print(signal_id)
            # sns.set_theme()
            df_aux = df_rearr[df_rearr["Signal_id"] == signal_id]

            # If the method has different parameters, add the parameters to the name.
            indexes = df_aux["Parameter"] != "{(),{}}"
            method_and_params = (
                df_aux.loc[indexes, "Method"] + "-" + df_aux.loc[indexes, "Parameter"]
            )
            df_aux.loc[indexes, "Method"] = method_and_params

            # Filter methods using the string given:
            if filter_str is not None:
                if filter_crit == "all":
                    a = [np.all([j in i for j in filter_str]) for i in df_aux["Method"]]
                if filter_crit == "any":
                    a = [np.any([j in i for j in filter_str]) for i in df_aux["Method"]]

                method_and_params = df_aux.iloc[a, :]
                df_aux = method_and_params

            if bars == False or plot_type == "lines":
                self.get_snr_plot(df_aux, x="SNRin", y="QRF", hue="Method", axis=ax)
            else:
                self.get_snr_plot_bars(
                    df_aux,
                    x="SNRin",
                    y="QRF",
                    hue="Method",
                    errbar_fun=errbar_fun,
                    errbar_params=errbar_params,
                    axis=ax,
                )

            if self.benchmark.task == "denoising" and magnitude == "difference":
                ax.set_ylabel("QRF - SNR")

            if self.benchmark.task == "detection":
                ax.set_ylabel("Detection Power")
                ax.set_ylim([0, 1.2])

            ax.grid(linewidth=0.1)
            ax.set_axisbelow(True)
            ax.set_title(signal_id)
            ax.legend(loc="upper left", frameon=False, fontsize="small")
            fig.set_size_inches(size)

            # Save figures to file.
            if savetofile:
                if path is None:
                    filename = os.path.join(
                        "results", self.task, "figures", "plot_" + signal_id + ".pdf"
                    )

                else:
                    filename = os.path.join(path, "plot_" + signal_id + ".pdf")

                fig.savefig(filename, bbox_inches="tight")

            list_figs.append(fig)
            ax = None  # Create new figure in the next iteration of the loop.
        return list_figs

    # --------------------------------------------------------------------------------------
    def get_summary_plotlys(
        self, df=None, bars=True, difference=False, varfun=None, ylabel=None, idx=0
    ):
        """Generates interactive plots with plotly.

        Returns:
            list : A list with plotly figures.
        """

        if df is None:
            df = self.benchmark.get_results_as_df()
            if type(df) == list:
                df = df[idx]

        if difference:
            for col in df.columns.values[4::]:
                df[col] = df[col] - col

        figs = []

        # Get dataframes with means and some variability measure (varfun)
        dfs_means = self._get_df_means(df=df)
        dfs_std_upper, dfs_std_lower = self._get_df_std(df=df, varfun=varfun)

        for signal_id, df_means, df_std_upper, df_std_lower in zip(
            self.signal_ids, dfs_means, dfs_std_upper, dfs_std_lower
        ):
            df3 = df_means.set_index("Method + Param").stack().reset_index()
            df3.rename(columns={"level_1": "SNRin", 0: "QRF"}, inplace=True)
            df3_std = df_std_upper.set_index("Method + Param").stack().reset_index()
            df3_std.rename(columns={"level_1": "SNRin", 0: "std"}, inplace=True)
            df3_std_minus = (
                df_std_lower.set_index("Method + Param").stack().reset_index()
            )
            df3_std_minus.rename(columns={"level_1": "SNRin", 0: "std"}, inplace=True)
            df3["std"] = df3_std["std"]
            df3["std-minus"] = df3_std_minus["std"]
            # print(df3)

            if bars:
                fig = px.bar(
                    df3,
                    x="SNRin",
                    y="QRF",
                    color="Method + Param",
                    #  markers=True,
                    barmode="group",
                    error_x="SNRin",
                    error_y="std",
                    error_y_minus="std-minus",
                    title=signal_id,
                )
            else:
                fig = px.line(
                    df3,
                    x="SNRin",
                    y="QRF",
                    color="Method + Param",
                    markers=True,
                    error_x="SNRin",
                    error_y="std",
                    error_y_minus="std-minus",
                    title=signal_id,
                )

            fig.update_layout(xaxis_title="SNRin (dB)")

            if ylabel is not None:
                fig.update_layout(yaxis_title=ylabel)

            figs.append(fig)

        return figs

    def get_html_figures(
        self,
        df=None,
        path=None,
        bars=True,
        difference=False,
        varfun=None,
        ylabel=None,
        idx=0,
    ):
        """
        Generate .html interactive plots files with plotly to show online.

        """
        if df is None:
            df = self.benchmark.get_results_as_df()
            if type(df) == list:
                df = df[idx]

        figs = self.get_summary_plotlys(
            df=df, bars=bars, difference=difference, varfun=varfun, ylabel=ylabel
        )

        for signal_id, fig in zip(self.signal_ids, figs):

            if path is None:
                path = "results"

            # Check path availability, otherwise create folder.
            if not os.path.exists(path):
                os.makedirs(path)

            filename = os.path.join(path, "plot_" + signal_id + ".html")
            with open(filename, "w") as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        return True

    def get_csv_files(self, path, idx=0):
        """Generates a table of mean results to .md file.
        Saves a .csv file with the results per signal.

        Returns:
            str: String containing the table.
        """

        # TODO Deal with results for different perf. metrics here!
        df = self.benchmark.get_results_as_df()
        if type(df) == list:
            df = df[idx]

        # Save a .csv with all the results.
        df_aux = self.rearrange_data_frame()
        filename = os.path.join(path, "results.csv")
        df_aux.to_csv(filename)

        # Save one .csv per signal.
        for signal_id in self.signal_ids:
            # Generate DataFrame with only signal information
            df2 = df[df["Signal_id"] == signal_id]

            # # Save .csv file for the signal.
            if path is None:
                path = "results"

            # Check path availability, otherwise create folder.
            if not os.path.exists(path):
                os.makedirs(path)

            filename = os.path.join(path, "results_" + signal_id + ".csv")
            df2.to_csv(filename)

        return True

    def elapsed_time_summary(self):
        """Get a DataFrame summarizing the elapsed times of the methods.

        Returns:
            DataFrame: Pandas DataFrame with a summary of the elapsed times.
        """
        mydict = self.benchmark.elapsed_time
        auxdic = {}

        for signal_id in mydict:
            for k in mydict[signal_id]:
                for k2 in mydict[signal_id][k]:
                    auxdic[signal_id + "-" + k + "-" + k2] = pd.DataFrame(
                        mydict[signal_id][k][k2]
                    )

        df3 = pd.concat(auxdic, axis=0)
        df3 = df3.reset_index()
        df3 = df3.drop(labels="level_1", axis=1)
        methid = np.unique(df3["level_0"])
        auxdic = {}
        for i in methid:
            auxdic[i] = (
                np.mean(df3[0][df3["level_0"] == i]),
                np.std(df3[0][df3["level_0"] == i]),
            )

        df = pd.DataFrame(auxdic)
        df = df.transpose()
        df.columns = ("Average time (s)", "Std")

        return df


# Use this function only for the CP CI shown in the interactive figures using Plotly:
def _clopper_pearson(x, alpha=0.05, bonferroni=1):
    """
    Clopper-Pearson confidence interval for Bernoulli parameter
    alpha: confidence level
    k: number of successes
    n: number of observations
    """
    alpha = alpha / bonferroni
    n = len(x)  # k: number of successes
    k = sum(x)  # n: number of observations
    lb = spst.beta.ppf(alpha / 2, k, n - k + 1)
    ub = spst.beta.ppf(1 - alpha / 2, k + 1, n - k)

    if np.isnan(ub):
        ub = 1.0

    return np.mean(x) - lb, ub - np.mean(x)


def _ci(x, alpha=0.05, bonferroni=1):
    significance = alpha / bonferroni * 100
    lb, ub = sns.utils.ci(sns.algorithms.bootstrap(x), which=100 - significance)
    return np.mean(x) - lb, ub - np.mean(x)
