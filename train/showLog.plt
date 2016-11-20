#!/usr/bin/gnuplot -c

# Run as
# ./showLog.plt /pathOfError/error.log

# Enable auto legend
set key autotitle columnhead
# Set tab as separator
set datafile separator tab
# Switch on the grid
set grid

plot ARG1 u :1 w lp lw 2, '' u :2 w lp lw 2
pause -1
