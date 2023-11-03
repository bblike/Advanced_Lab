from tkinter import *
from tkinter import ttk
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk

start_point_0 = 0
mid_point_0 = 0
start_point_1 = 0
mid_point_1 = 0


def fresh_window(e11, e12, e21, e22):
    start_point1_0 = e11.get()
    mid_poin_0 = e12.get()
    start_point_1 = e21.get()
    mid_point_1 = e22.get()


def create_window(myImage):
    root = Tk()
    root.title("select the analyse points")
    root.geometry('800x400')

    im = Image.open(myImage)
    photo = ImageTk.PhotoImage(im.resize([400,400]))
    imlabel = Label(image=photo)
    imlabel.grid(row=0, column=3, columnspan=1, rowspan=5, sticky=W + N)
    Label(root, text="start point").grid(row=1, column=0)
    Label(root, text="mid point").grid(row=2, column=0)

    fresh = ttk.Frame(root, padding=0)
    e11 = Entry(root)
    e12 = Entry(root)
    e21 = Entry(root)
    e22 = Entry(root)
    yes = ttk.Frame(root, padding=0)
    e11.grid(row=1, column=1)
    e12.grid(row=1, column=2)
    e21.grid(row=2, column=1)
    e22.grid(row=2, column=2)
    fresh.grid(row=3, column=1)
    yes.grid(row=4, column=1)
    ttk.Button(fresh, text="fresh", command=fresh_window(e11, e12, e21, e22)).grid(row=3, column=1)
    ttk.Button(yes, text="Yes", command=root.destroy).grid(row=4, column=1)

    p1 = Canvas()

    root.mainloop()
