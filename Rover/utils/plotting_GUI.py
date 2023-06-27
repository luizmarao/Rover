import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog, messagebox
import os
from os import path as osp
import csv
import re
import numpy as np
import time





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

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class PlotArea(tk.Frame):

    def __init__(self, parent, controller, progress_files):
        self.set_progress_files(progress_files)
        tk.Frame.__init__(self, parent)

        self.line_plot_list = []
        self.figure = Figure(figsize=(10, 5))
        self.ax = self.figure.add_subplot(111)

        separator = ttk.Separator(self, orient='horizontal')
        separator.pack(anchor=tk.NW, fill=tk.X)
        title_label = tk.Label(self, text='Plot Title: ')
        self.title_field = tk.Entry(self)
        title_label.pack(anchor=tk.NW, pady=5)
        self.title_field.pack(anchor=tk.NW, fill=tk.X, pady=5, padx=2)
        separator = ttk.Separator(self, orient='horizontal')
        separator.pack(anchor=tk.NW, fill=tk.X)
        fields_label = tk.Label(self, text='Select Fields and smooth: ')
        fields_label.pack(anchor=tk.NW, pady=5)
        self.plot_config_area = tk.Frame(self)
        self.plot_config_area.field_boxes = {}
        self.plot_config_area.smooth_size = None

        self.plot_config_area.pack(anchor=tk.NW, fill=tk.BOTH)

        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH)


    def update_plot_config_area(self):
        for field, box in self.plot_config_area.field_boxes.items():
            box.destroy()
        self.plot_config_area.field_boxes.clear()
        fields_per_row = 8
        if not len(self.progress_files) == 0:
            field_boxes = {}
            with open(self.progress_files[0], 'r') as progress_file:
                csv_reader = csv.DictReader(progress_file, delimiter=',')

                fields = list(next(csv_reader).keys())
                default_fields = ["rollout/success_rate (%)", "rollout/ep_rew_mean", "rollout/ep_rew_min",
                                  "rollout/ep_rew_max", "rollout/Avg÷MEA (%)"]
                fields.sort()
                for idx, field in enumerate(fields):
                    box_var = tk.IntVar()
                    box = tk.Checkbutton(self.plot_config_area, text=field, onvalue=1, offvalue=0, variable=box_var)
                    box.var = box_var
                    if field in default_fields:
                        box.select()

                    box.configure(command=self.plot_update)
                    box.grid(column=idx % fields_per_row, row=int(idx / fields_per_row), sticky=tk.W)
                    field_boxes[field] = box

            if self.plot_config_area.smooth_size is None:
                self.plot_config_area.smooth_size = tk.Spinbox(self.plot_config_area, from_=1, to=50, wrap=False)
                tk.Label(self.plot_config_area, text='Smooth Window:').grid(column=fields_per_row, row=0, sticky=tk.W)
                self.plot_config_area.smooth_size.grid(column=fields_per_row, row=1)
                current_value = tk.StringVar(value='20')
                self.plot_config_area.smooth_size.configure(command=self.plot_update, textvariable=current_value)

            self.plot_config_area.field_boxes = field_boxes

    def set_progress_files(self, progress_files):
        self.progress_files = progress_files
        self.file_watchers = [FileWatcher(progress_file_path) for progress_file_path in progress_files]


    def add_progress_file(self, progress_file):
        self.progress_files += [progress_file]
        self.file_watchers += [FileWatcher(progress_file)]


    def plot_update(self):
        legends = []
        self.ax.clear()
        self.figure.legends.clear()
        for progress_file_path in self.progress_files:
            with open(progress_file_path, 'r') as progress_file:
                csv_reader = csv.DictReader(progress_file, delimiter=',')

                fields = list(next(csv_reader).keys())

                xlabel = 'time/total_timesteps'
                smooth_window = int(self.plot_config_area.smooth_size.get())
                selected_fields = []
                for field, box in self.plot_config_area.field_boxes.items():
                    if box.var.get():
                        selected_fields.append(field)
                is_selecting_fields = False
                solid_line_re = re.compile('-')
                dashed_line_re = re.compile('--')
                dotted_line_re = re.compile(r'\*')
                dash_dotted_line_re = re.compile(r'\*-')
                line_types = {selected_fields[-1]: '-'}
                title = self.title_field.get()
                xlabel_custom = ''
                ylabel = ''
                xmax = 0

                yvalues = {f: [] for f in selected_fields}
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
                    xvalues = xvalues[smooth_window - 1:]
                last_index = 0
                first_time = True

                for i, sf in enumerate(selected_fields):
                    if xmax == 0:
                        if first_time:
                            aux_plot, = self.ax.plot(xvalues, yvalues[sf], label='name')
                            self.line_plot_list.append(aux_plot)
                        else:
                            self.line_plot_list[i].set_ydata(yvalues[sf])
                            self.line_plot_list[i].set_xdata(xvalues)
                            max_y = np.max([max_y, np.max(yvalues[sf])])
                            min_y = np.min([min_y, np.min(yvalues[sf])])
                    else:
                        if first_time:
                            aux_plot = self.ax.plot(xvalues[:last_index], yvalues[sf][:last_index], label='name')
                            self.line_plot_list.append(aux_plot)
                        else:
                            self.line_plot_list[i].set_ydata(yvalues[sf][:last_index])
                            self.line_plot_list[i].set_xdata(xvalues[:last_index])
                            max_y = np.max([max_y, np.max(yvalues[sf][:last_index])])
                            min_y = np.min([min_y, np.min(yvalues[sf][:last_index])])
                if first_time:
                    self.ax.ticklabel_format(style='sci', scilimits=(0, 3))
                    self.ax.grid(True)
                    self.ax.set_xlabel(xlabel) if xlabel_custom == '' else self.ax.set_xlabel(xlabel_custom)
                    if not ylabel == '':
                        self.ax.set_ylabel(ylabel)
                    self.ax.set_title(title)
                    if len(self.progress_files) > 1:
                        extra_legend = progress_file_path.split('/')[-2].split('-')[0] + '/'
                        legends += [extra_legend + legend for legend in selected_fields.copy()]
                    else:
                        legends += selected_fields.copy()
                    # self.figure.legend(legends).set_draggable(True)
                    # plt.ion()
                    # self.figure.show()
                else:
                    self.ax.set_xlim(0, xvalues[-1] * 1.01 if xmax == 0 else xvalues[:last_index][-1] * 1.01)
                    self.ax.set_ylim(min_y * 1.01, max_y * 1.01)
                    #self.ax.draw()
                    # plt.pause(0.001)
                    # time.sleep(1.)
                    #self.figure.canvas.flush_events()
                first_time = False
        self.figure.legend(legends).set_draggable(True)
        self.canvas.draw()

class App(tk.Tk):
    num_tree_items = 0

    def __init__(self):
        super().__init__()
        self.title("Rover Plotting Tool")
        self.grid()
        self.resizable(True, True)
        self.create_main_window()

    def create_exp_selector_side(self):
        left_frame = ttk.Frame(self, height=500, width=200)
        left_frame.columnconfigure(3, weight=6)

        tree_label = ttk.Label(left_frame, text="Folders")
        tree_label.grid(column=0, row=0, sticky=tk.W)
        ttk.Label(left_frame, text="          ").grid(column=1, row=0, sticky=tk.W)
        ttk.Label(left_frame, text="             ").grid(column=2, row=0, sticky=tk.W)

        add_button = ttk.Button(left_frame, text="+", command=self.add_folder, width=1)
        add_button.grid(column=2, row=0, sticky=tk.E)

        remove_button = ttk.Button(left_frame, text="-", command=self.remove_folder, width=1)
        remove_button.grid(column=3, row=0, sticky=tk.W)

        tree = ttk.Treeview(left_frame, selectmode="browse", height=40)
        tree.grid(column=0, row=1, sticky=tk.NSEW, columnspan=4)

        tree_vscroll = ttk.Scrollbar(left_frame, orient='vertical', command=tree.yview())
        tree_vscroll.grid(column=4, row=1, sticky='NSW')

        tree['yscrollcommand'] = tree_vscroll.set

        tree_hscroll = ttk.Scrollbar(left_frame, orient='horizontal', command=tree.xview())
        tree_hscroll.grid(column=0, row=2, sticky='WE', columnspan=4)

        tree['xscrollcommand'] = tree_hscroll.set
        self.tree = tree

        plot_button = ttk.Button(left_frame, text='Plot Selected', command=self.plot_selected)
        plot_button.grid(column=0, row=3, sticky=tk.EW, columnspan=2)

        add_plot_button = ttk.Button(left_frame, text='Add to Plot', command=self.add2plot)
        add_plot_button.grid(column=2, row=3, sticky=tk.EW, columnspan=3)

        return left_frame

    def create_plot_side(self):
        width = 1600
        heigth = 800
        right_frame = ttk.Frame(self, height=heigth, width=width)
        right_frame.grid(column=2, row=0, sticky=tk.NSEW)

        self.tab_control = ttk.Notebook(right_frame, height=heigth, width=width)
        self.tab_control.grid(column=0, row=0, sticky=tk.NSEW)

        tab1 = PlotArea(self.tab_control, self, progress_files=[])
        self.tab_plus = tk.Frame(self.tab_control)
        self.tab_control.add(tab1, text="Plot 1")
        self.tab_control.add(self.tab_plus, text="+")
        self.tab_control.bind('<<NotebookTabChanged>>', self.new_tab)
        return right_frame

    def new_tab(self, *args):
        tab_name = self.tab_control.select()
        if tab_name.split('.')[-1] == '!frame':
            num_tabs = len(self.tab_control.children)
            tab = PlotArea(self.tab_control, self, progress_files=[])
            plot_name = simpledialog.askstring('Create New Tab', 'New Tab\'s Name:', initialvalue='Plot' + str(num_tabs))
            self.tab_control.insert(self.tab_plus, tab, text=plot_name)
            self.tab_control.select(num_tabs - 1)

    def close_tab(self):
        num_tabs = len(self.tab_control.children)
        if num_tabs > 2:
            tab_name = self.tab_control.select()
            plot_area = self.tab_control.children[tab_name.split('.')[-1]]
            self.tab_control.select(0)
            self.tab_control.forget(tab_name)
            plot_area.destroy()
        else:
            messagebox.showwarning(title='Warning', message='Can\'t remove last tab')

    def rename_tab(self):
        tab_name = self.tab_control.select()
        plot_name = {'text': simpledialog.askstring('Rename Tab', 'New Tab\'s Name:', initialvalue='')}
        self.tab_control.tab(tab_name, **plot_name)

    def create_main_window(self):
        # Create the menu bar
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Close Current Plot", command=self.close_tab)
        file_menu.add_command(label="Rename Current Plot", command=self.rename_tab)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        menu_bar.add_cascade(label="File", menu=file_menu)

        # Create left frame with tree view
        left_frame = self.create_exp_selector_side()
        left_frame.grid(column=0, row=0, sticky=tk.NS)


        separator = ttk.Separator(self, orient='vertical')
        separator.grid(column=1, row=0, sticky=tk.NS)

        # Create right frame with tab panel
        right_frame = self.create_plot_side()
        right_frame.grid(column=2, row=0, sticky=tk.NS)

    def add_folder(self):
        # Open file dialog to select folder
        folder_path = filedialog.askdirectory()

        if isinstance(folder_path, str):
            # Add folder to tree view
            self.tree.insert("", "end", text=folder_path, iid=self.num_tree_items, values=(), open=True)
            root_folder_iid = self.num_tree_items
            self.num_tree_items += 1
            folder_content = os.listdir(folder_path)
            exp_folders = []
            for content in folder_content:
                content_path = osp.join(folder_path, content)
                if osp.isdir(content_path):
                    if 'progress.csv' in os.listdir(content_path):
                        exp_folders.append(content)
                        self.tree.insert(root_folder_iid, "end", text=content, iid=self.num_tree_items, values=())
                        self.num_tree_items += 1

    def remove_folder(self):
        # Remove selected item from tree view
        if not len(self.tree.selection()) == 0:
            selected_item = self.tree.selection()[0]
            self.tree.delete(selected_item)

    def plot_selected(self):
        if not len(self.tree.selection()) == 0:
            is_parent_node = [x in self.tree.get_children() for x in self.tree.selection()]
            if sum(is_parent_node) == 0:  # root exp folders have no exp to plot
                tab_name = self.tab_control.select()
                plot_area = self.tab_control.children[tab_name.split('.')[-1]]
                progress_files = [osp.join(self.tree.item(self.tree.parent(idx))['text'], self.tree.item(idx)['text'],
                                           'progress.csv') for idx in self.tree.selection()]
                plot_area.set_progress_files(progress_files)
                plot_area.update_plot_config_area()
                plot_area.plot_update()

    def add2plot(self):
        if not len(self.tree.selection()) == 0:
            is_parent_node = [x in self.tree.get_children() for x in self.tree.selection()]
            if sum(is_parent_node) == 0:  # root exp folders have no exp to plot
                tab_name = self.tab_control.select()
                plot_area = self.tab_control.children[tab_name.split('.')[-1]]
                current_exps = [exp.split('/')[-2] for exp in plot_area.progress_files]
                for idx in self.tree.selection():
                    if self.tree.item(idx)['text'] not in current_exps:
                        plot_area.add_progress_file(osp.join(self.tree.item(self.tree.parent(idx))['text'],
                                                             self.tree.item(idx)['text'], 'progress.csv'))

                plot_area.plot_update()


if __name__ == "__main__":
    app = App()
    app.mainloop()
