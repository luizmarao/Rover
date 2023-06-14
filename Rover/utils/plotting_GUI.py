import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from os import path as osp


class PlotArea(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])

        canvas = FigureCanvasTkAgg(f, self)
        # canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class App(tk.Tk):
    num_tree_items = 0

    def __init__(self):
        super().__init__()
        #self.geometry("900x700")
        self.title("Rover Plotting Tool")
        self.resizable(True, True)
        self.create_main_window()

    def create_exp_selector_side(self):
        left_frame = ttk.Frame(self)
        # left_frame.columnconfigure(0, weight=1)

        tree_label = ttk.Label(left_frame, text="Folders")
        tree_label.grid(column=0, row=0, sticky=tk.W)

        # buttons_frame = ttk.Frame(left_frame)
        # buttons_frame.grid(column=1, row=0, sticky=tk.W)
        add_button = ttk.Button(left_frame, text="+", command=self.add_folder)
        add_button.grid(column=1, row=0, sticky=tk.W)

        remove_button = ttk.Button(left_frame, text="-", command=self.remove_folder)
        remove_button.grid(column=2, row=0, sticky=tk.W)

        button3 = ttk.Button(left_frame, text='3')
        button3.grid(column=3, row=0, sticky=tk.W)
        button4 = ttk.Button(left_frame, text='4')
        button4.grid(column=4, row=0, sticky=tk.W)
        button5 = ttk.Button(left_frame, text='5')
        button5.grid(column=5, row=0, sticky=tk.W)
        button6 = ttk.Button(left_frame, text='6')
        button6.grid(column=6, row=0, sticky=tk.W)

        tree = ttk.Treeview(left_frame, selectmode="browse")
        tree.grid(column=0, row=1, sticky=tk.NSEW, columnspan=7)

        tree_scroll = ttk.Scrollbar(left_frame, orient='vertical', command=tree.yview())
        tree_scroll.grid(column=7, row=1, sticky='NSW')

        tree['yscrollcommand'] = tree_scroll.set
        self.tree = tree

        return left_frame

    def create_plot_side(self):
        right_frame = ttk.Frame(self)
        right_frame.grid(column=2, row=0, sticky=tk.NSEW)

        tab_control = ttk.Notebook(right_frame)
        tab_control.grid(column=0, row=0)

        # tab1 = ttk.Frame(tab_control)
        tab1 = PlotArea(tab_control, self)
        tab_control.add(tab1, text="Plot 1")

        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text="Plot 2")

        return right_frame

    def create_main_window(self):
        # Create the menu bar
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open")
        file_menu.add_command(label="Save")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        menu_bar.add_cascade(label="File", menu=file_menu)

        # Create the paned window
        #paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=4, sashrelief=tk.RAISED)
        #paned_window.grid(column=0, row=0)

        # Create left frame with tree view
        left_frame = self.create_exp_selector_side()
        left_frame.grid(column=0, row=0, sticky='NS')


        separator = ttk.Separator(self, orient='vertical')
        separator.grid(column=1, row=0, sticky=tk.NS)

        # Create right frame with tab panel
        right_frame = self.create_plot_side()
        right_frame.grid(column=1, row=0, sticky=tk.NS)


    def open_file(self):
        # Placeholder function for "Open" command
        pass

    def add_folder(self):
        # Open file dialog to select folder
        folder_path = filedialog.askdirectory()

        # Add folder to tree view
        self.tree.insert("", "end", text=folder_path, iid=self.num_tree_items, values=(), open=True)
        root_folder_iid = self.num_tree_items
        self.num_tree_items += 1
        folder_content = os.listdir(folder_path)
        exp_folders = []
        for content in folder_content:
            if osp.isdir(osp.join(folder_path, content)):
                exp_folders.append(content)
                self.tree.insert(root_folder_iid, "end", text=content, iid=self.num_tree_items, values=())
                self.num_tree_items += 1

    def remove_folder(self):
        # Remove selected item from tree view
        selected_item = self.tree.selection()[0]
        self.tree.delete(selected_item)


if __name__ == "__main__":
    app = App()
    app.mainloop()
