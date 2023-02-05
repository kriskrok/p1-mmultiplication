import random
import time
from os import getpid
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psutil
from deprecated import deprecated


def populate(n: int, m: int, seed: int = 42):
    """ Create n X m size matrix populated with random numbers
    Parameters
    ----------
    n : int
        The amount of rows in the matrix
    m : int
        The amount of columns in the matrix
    seed: int
        Seed value for random

    Returns
    -------
    List[List[float]]
        Matrix prepopulated with random numbers in the range of (0,1)"""
    random.seed(seed)

    # x for column length, y for row height
    matrix = [[random.uniform(0, 1) for x in range(0, m)] for y in range(0, n)]

    print('done populating')

    return matrix


def multiply(A: List[List[float]] = [], B: List[List[float]] = []) -> List[List[float]]:
    """ Calculates matrix product for two matrix's
    Parameters
    ----------
    A : List[List[float]]
        Matrix A, the number of columns must be equal to the number of rows in matrix B.
    B : List[List[float]]
        Matrix B, the number of rows must be equal to the number of columns in matrix A

    Returns
    -------
    List[List[float]]
        Matrix product AB"""

    # check for compatibility
    if len(A) > 0 and 0 < len(B) == len(A[0]):
        rows = len(A)  # height
        columns = len(B[0])  # width

        # initialize result matrix C
        C = [[0 for x in range(0, columns)] for y in range(0, rows)]

        # row of A and a column of B.
        for count, row in enumerate(range(rows), start=0):  # default is zero
            if count % 50 == 0:
                print(f"Currently on row {row}")
            #        for row in range(rows): # loop through rows of A
            for col in range(columns):  # loop through columns of B
                for element in range(len(B)):  # access elements in selected column
                    C[row][col] += A[row][element] * B[element][col]

            A[count] = None
        # premature optimisation
        return C


def init():
    """Control logic for matrix multiplication assignment"""
    scale = int(10**3)

    A = populate(scale, 10 ** 3)  # 8000?
    B = populate(10 ** 3, scale)  # 8000?

    temp = multiply(A, B)
    # clean up unneeded matrix's
    del A, B

    C = populate(scale, 1)
    # Compute result matrix D
    D = multiply(temp, C)


@deprecated(reason="Usage replaced by psrecord")
def memory():
    """For additional information, see
    https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_info#psutil.Process.memory_info.
    -------
    All memory info values expressed in bytes"""

    print(psutil.Process().memory_info())
    print(psutil.Process().memory_info().rss / (1024 * 1024))  # MB


def plot_matrix(A: List[List[float]]):
    """ Plots a cumulative distribution function (CDF) for a given matrix
    Parameters
    ----------
    A : List[List[float]]
        Matrix to be plotted."""

    # Compute the histogram of A.
    count, bin_edges = np.histogram(A, bins=50, density=False)

    # find the PDF of the histogram with count values
    pdf = count / sum(count)

    # calculate the CDF
    cdf = np.cumsum(pdf)

    # plotting probability distribution function PDF and cumulative distribution function CDF
    plt.plot(bin_edges[1:], cdf, color="c", label="CDF", marker="|")
    plt.plot(bin_edges[1:], pdf, color="orange", label="PDF", marker="|")
    plt.legend(loc=5)
    plt.xlabel('integer values')
    plt.ylabel('frequency')

    plt.title('Cumulative distribution function for matrix A')
    plt.show()


@deprecated(reason="Usage replaced by psrecord")
def plot_usage_from_pidstat_results():
    """ Plots CPU utilization and memory utilization graph from data captured by pidstat."""

    cpu, rss = [], []
    with open("trimmed_results.txt") as file:
        for row in file:
            row = row.replace("\n", "")
            parts = row.split(' ')
            cpu.append(float(parts[0]))
            rss.append(int(parts[1]))

    # plot the graph
    fig, ax = plt.subplots()
    # Twin the x-axis twice to make two independent y-axes.
    xtwin = ax.twinx()

    # RSS
    line1 = ax.plot_matrix(rss, label="RSS", c="Blue")
    ax.set_ylabel("RSS y-label")
    ax.tick_params(axis='y', color='Green')

    # CPU
    line2 = xtwin.plot_matrix(cpu, label="CPU", c="Green")
    xtwin.set_ylabel("CPU utilization (%)")
    xtwin.tick_params(axis='y', color='Green')

    # both labels to same legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc=8)
    ax.set_xlabel('time (s)')

    plt.title('CPU usage and RSS (Resident Set Size) in kilobytes')
    plt.show()


if __name__ == '__main__':
    print(f"Running as process {getpid()}")
    start = time.perf_counter()
    init()
    finish = time.perf_counter()
    print(f"Finished in {finish - start} seconds")
