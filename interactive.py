from tkinter import *
from tkinter import ttk
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk


def create_window(myImage):
    root = Tk()
    root.title("select the analyse points")
    root.geometry('600x600')

    im = Image.open(myImage)
    photo = ImageTk.PhotoImage(im)
    imlabel = Label(image=photo)
    imlabel.grid(row=0, column=2, columnspan=2, rowspan=2, sticky = S+N)
    Label(root, text="start point").grid(row=1)
    Label(root, text="mid point").grid(row=2)

    e1 = Entry(root)
    e2 = Entry(root)
    yes = ttk.Frame(root, padding=10)
    e1.grid(row=1, column=1)
    e2.grid(row=2, column=1)
    yes.grid()
    ttk.Button(yes, text = "Yes", command=root.destroy).grid(row=3, column=1)



    root.mainloop()




