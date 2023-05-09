# import csv
#     file_data = self.request.get('file_in')
#     file_data_list = file_data.split('\n')
#     file_Reader = csv.reader(file_data_list)
#     for fields in file_Reader:
#         print row
import getopt
import sys
from pathlib import Path
import csv
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import time
import os

#idea from https://stackoverflow.com/a/182259
class FileWatcher:
    def __init__(self, file_path):
        self._cached_stamp = 0
        self.filename = file_path

    def has_news(self):
        stamp = os.stat(self.filename).st_mtime
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        else:
            return False

# from here: https://stackoverflow.com/a/14314054/6609908
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

BLACK       = '\033[30m'
RED         = '\033[31m'
GREEN       = '\033[32m'
ORANGE	    = '\033[33m'
BLUE	    = '\033[34m'
MAGENTA	    = '\033[35m'
CYAN	    = '\033[36m'
LIGHTGRAY	= '\033[37m'
YELLOW	    = '\033[93m'
DEFAULT	    = '\033[39m'
ENDCOLOR    = '\033[0m'

help_message = '''
usage: plot.py  <spinup progress file>
                [-h]
                [--xaxis XAXIS]
                [--COLUMN_NAME]
                [--smooth SMOOTH]
                [--title TITLE]
                [--xlabel XLABEL]
                [--ylabel YLABEL]
                [--xmax XMAX]
                
optional arguments:
    -h, --help                  show this help message and exit
    --xaxis XAXIS, -x XAXIS     where XAXIS must be one of the fields of the spinup progress file
    --COLUMN_NAME[-|--|*|*-]    where COLUMN_NAME must be one of the fields of the spinup progress file
                                append - for solid (default), -- for dashed, * for dotted or *- for dash-dotted 
    --smooth SMOOTH, -s SMOOTH  where SMOOTH is the window size (must be > 1) of the moving average
    --title TITLE, -t TITLE     where TITLE is the title of the plot
    --xlabel XLABEL             where XLABEL is the label of the x axis (default: xaxis name)
    --ylabel YLABEL             where YLABEL is the label of the y axis (default: none)
    --xmax XMAX                 where XMAX is the amount of points that will be plotted (default: plots all)
    -l, --list-columns          lists all columns in the file and exit
'''

def print_colored(color, what):
    print(color + what + ENDCOLOR)

print_help = False

if len(sys.argv) == 1:
    print(help_message)
    exit(0)
elif sys.argv[1] in ('-h', '--h', '--he', '--hel', '--help'):
    print(help_message)
    exit(0)

try:
    progress_file_path = Path(sys.argv[1])
except:
    print_colored('First argument should be a valid spinup progress file.')
    exit(1)

if not progress_file_path.is_absolute():
    progress_file_path = Path.cwd() / progress_file_path

if not progress_file_path.is_file():
    print_colored('First argument should be a valid spinup progress file.')
    exit(1)

argv = sys.argv[2:]

first_time = True

file_watcher = FileWatcher(progress_file_path.resolve())

figure = None
ax = None
line_plot_list = []

while(True):
    with open(progress_file_path, 'r') as progress_file:
        csv_reader = csv.DictReader(progress_file, delimiter=',')

        fields = list(next(csv_reader).keys())

        try:
            options, args = getopt.getopt(argv, 'x:s:t:l', longopts=['smooth=', 'title=', 'xaxis=', 'xlabel=', 'ylabel=', 'xmax='] +
                                                                   fields +
                                                                   [f+'-' for f in fields] +
                                                                   [f+'--' for f in fields] +
                                                                   [f+'*' for f in fields] +
                                                                   [f+'*-' for f in fields])
        except getopt.GetoptError as e:
            print_colored(RED, 'Options must be one of the following:')
            print_colored(RED, '--list-columns, -xaxis XAXIS, --smooth WINDOW_SIZE, --title TITLE, --xlabel XLABEL, --ylabel YLABEL, --xmax XMAX' + ', --'.join(fields))
            exit(1)

        xlabel = 'time/total_timesteps'
        smooth_window = 1
        selected_fields = ['eprewmean']
        is_selecting_fields = False
        solid_line_re = re.compile('-')
        dashed_line_re = re.compile('--')
        dotted_line_re = re.compile(r'\*')
        dash_dotted_line_re = re.compile(r'\*-')
        line_types = {selected_fields[-1]: '-'}
        title = ''
        xlabel_custom = ''
        ylabel = ''
        xmax = 0

        for option, value in options:
            if option in ('-l', '--list-columns'):
                print("Available columns in ", progress_file_path, ":")
                print(', '.join(fields))
                print()
                exit(0)
            if option in ('-x', '--xaxis'):
                if not value in fields:
                    print_colored(YELLOW, '-x should be one of the following:\n{}'.format(', '.join(fields)))
                    print_colored(RED, 'Reseting -x to default: ' + xlabel)
                else:
                    xlabel = value
            elif option in ('-s', '--smooth'):
                try:
                    smooth_window = int(value)
                except ValueError as e:
                    print('--smooth (or -s) value error: ' + str(e))
                    print_colored(RED, 'Ignoring --smoothing (or -s)')
                    smooth_window = 1
                if smooth_window <= 1:
                    print_colored(YELLOW, '--smooth (or -s) should have value greater than 1')
                    print_colored(RED, 'Ignoring --smoothing (or -s)')
                    smooth_window = 1
            elif option[2:] in fields + [f+'-' for f in fields] + [f+'--' for f in fields] + [f+'*' for f in fields] + [f+'*-' for f in fields]:
                if not is_selecting_fields:
                    selected_fields = []
                    is_selecting_fields = True
                selected_fields.append(option[2:])
                if dashed_line_re.search(selected_fields[-1]):
                    selected_fields[-1] = selected_fields[-1][:-2]
                    line_types[selected_fields[-1]] = ('--')
                elif dash_dotted_line_re.search(selected_fields[-1]):
                    selected_fields[-1] = selected_fields[-1][:-2]
                    line_types[selected_fields[-1]] = ('*-')
                elif solid_line_re.search(selected_fields[-1]):
                    selected_fields[-1] = selected_fields[-1][:-1]
                    line_types[selected_fields[-1]] = ('-')
                elif dotted_line_re.search(selected_fields[-1]):
                    selected_fields[-1] = selected_fields[-1][:-1]
                    line_types[selected_fields[-1]] = ('*')
                else:
                    line_types[selected_fields[-1]] = ('-')
            elif option in ('-t', '--title'):
                title = value
            elif option == '--xlabel':
                xlabel_custom = value
            elif option == '--ylabel':
                ylabel = value
            elif option == '--xmax':
                try:
                    xmax = float(value)
                except ValueError as e:
                    print('--xmax value error: ' + str(e))
                    print_colored(RED, 'Ignoring --xmax')
                    xmax = 0


        yvalues = {f:[] for f in selected_fields}
        xvalues = []

        progress_file.seek(0)
        next(csv_reader)

        for row in csv_reader:
            xvalues.append(float(row[xlabel]))
            for sf in selected_fields:
                yvalues[sf].append(float(row[sf]))

    if smooth_window > 1:
        for sf in selected_fields:
            yvalues[sf] = moving_average(yvalues[sf], smooth_window)
        xvalues = xvalues[smooth_window - 1 : ]

    last_index = 0

    if xmax > xvalues[-1]:
        print_colored(YELLOW, 'Ignoring xmax option ({}) because it is greater than actual maximum x value ({})'.format(xmax, xvalues[-1]))
        xmax = 0
    else:
        for i, x in enumerate(xvalues):
            if x > xmax:
                last_index = i
                break

    if first_time:
        figure = plt.figure()
        ax = figure.add_subplot(111)

    max_y = -9999999999
    min_y = 9999999999

    for i, sf in enumerate(selected_fields):
        if xmax == 0:
            if first_time:
                aux_plot, = ax.plot(xvalues, yvalues[sf], line_types[sf], label='name')
                line_plot_list.append(aux_plot)
            else:
                line_plot_list[i].set_ydata(yvalues[sf])
                line_plot_list[i].set_xdata(xvalues)
                max_y = np.max([max_y, np.max(yvalues[sf])])
                min_y = np.min([min_y, np.min(yvalues[sf])])
        else:
            if first_time:
                aux_plot = ax.plot(xvalues[:last_index], yvalues[sf][:last_index], line_types[sf], label='name')
                line_plot_list.append(aux_plot)
            else:
                line_plot_list[i].set_ydata(yvalues[sf][:last_index])
                line_plot_list[i].set_xdata(xvalues[:last_index])
                max_y = np.max([max_y, np.max(yvalues[sf][:last_index])])
                min_y = np.min([min_y, np.min(yvalues[sf][:last_index])])

    if first_time:
        ax.ticklabel_format(style='sci',scilimits=(0,3))
        ax.grid(True)
        plt.xlabel(xlabel) if xlabel_custom == '' else plt.xlabel(xlabel_custom)
        if not ylabel == '':
            ax.ylabel(ylabel)
        ax.set_title(title)
        figure.legend(selected_fields).set_draggable(True)
        plt.ion()
        plt.show(block=False)
    else:                
        ax.set_xlim(0, xvalues[-1]*1.01 if xmax == 0 else xvalues[:last_index][-1]*1.01)
        ax.set_ylim(min_y*1.01, max_y*1.01)
        plt.draw()
        #plt.pause(0.001)
        time.sleep(1.)
        figure.canvas.flush_events()

    first_time = False

    while True:
        #print('stamp:',file_watcher._cached_stamp)
        if file_watcher.has_news() and not ('--xmax' in [option[0] for option in options]):
            break
        else:
            try:
                #plt.pause(0.2)
                figure.canvas.flush_events()
                time.sleep(1.)
            except:
                exit(0)
