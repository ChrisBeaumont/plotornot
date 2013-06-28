from flask import Flask, send_file, request, render_template
from matplotlib import rcParamsDefault, rc_context, rcParams, rcParamsOrig
import matplotlib.pyplot as plt
import numpy as np
from cStringIO import StringIO
from urllib import urlopen
import json
from random import shuffle
from glob import glob

app = Flask(__name__)

def get_styles():
    '''Return 2 random rcParams styles'''
    files = glob('params/*')
    shuffle(files)
    return [json.load(open(f)) for f in files[:2]]


def mpl_figure_data(f):
    data = StringIO()
    f.canvas.print_png(data)
    data = data.getvalue().encode('base64')
    return data

def color_brewer_cycles():
    return [{'axes.color_cycle': ['#E41A1C', '#377EB8', '#4DAF4A',
             '#984EA3', '#FF7F00']},
             {'axes.color_cycle': ['#7FC97F', '#BEAED4', '#FDC086',
              '#FDC086', '#FFF99', '#386CDO']}
            ]

def make_plot(style):
    with rc_context(style):
        rcParams['figure.figsize'] = 6,4
        rcParams['figure.dpi'] = 75
        rcParams['figure.facecolor'] = '#ffffff'
        plt.clf()
        plt.cla()

        np.random.seed(42)
        x1 = np.random.normal(0, 1, (500,))
        x2 = np.random.normal(3, 1, (300,))
        x3 = np.random.normal(7, 2, (500,))
        kwargs = dict(range=(-10, 20), alpha=.7)
        plt.hist(x1, 50, **kwargs)
        plt.hist(x2, 50, **kwargs)
        plt.hist(x3, 50, **kwargs)
        plt.xlabel('The x axis')
        plt.ylabel('The y axis')
        plt.gcf().set_facecolor('w')
        result = mpl_figure_data(plt.gcf())
    return result

@app.route("/")
def main():
    s1, s2 = get_styles()
    assert s1 != s2
    d1, d2 =  map(make_plot, (s1, s2))
    return render_template('main.html', image_1=d1, image_2=d2)


if __name__ == "__main__":
    app.run(debug=True)
