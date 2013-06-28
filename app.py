# coding: utf-8

""" Plot or Not! """

from __future__ import division, print_function

# Standard library
import os, sys
from flask import Flask, send_file, request, render_template, redirect
from matplotlib import rcParamsDefault, rc_context, rcParams, rcParamsOrig
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
import numpy as np
from cStringIO import StringIO
from urllib import urlopen
import json
import random
from glob import glob

# Third-party
from flask import Flask, send_file, request, render_template
from matplotlib import rcParamsDefault, rc_context, rcParams, rcParamsOrig
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from matplotlib.lines import Line2D
import numpy as np

app = Flask(__name__)

def get_styles():
    '''Return 2 random rcParams styles'''
    files = glob('params/*')
    random.shuffle(files)
    
    styles = []
    for f in files[:2]:
        this_rc = rcParamsDefault.copy()
        s = json.load(open(f))
        for k,v in s.items():
            this_rc[k] = v
        styles.append(this_rc)
    return styles


def mpl_figure_data(f):
    data = StringIO()
    f.canvas.print_png(data)
    data = data.getvalue().encode('base64')
    return data

def plot_generator():
    plot_function = random.choice(['plot', 'hist', 'contour'])

    N_points = 100
    N_datasets = np.random.randint(2, 8)
    data = []
    for ii in range(N_datasets):
        if plot_function == 'plot':
            x = np.linspace(0., 10., N_points)
            A = np.random.uniform(-5., 5)
            B = np.random.uniform(-5., 5)
            y_func = random.choice([lambda x: 0.1*A*x + B,
                                    lambda x: 0.1*A*x**2 + B,
                                    lambda x: A*np.sin(B*x),
                                    lambda x: A*np.cos(B*x)])
            y = y_func(x)
            d = (x, y)
        elif plot_function == 'hist':
            d = (np.random.normal(np.random.uniform(0, N_datasets*5),
                                  np.random.uniform(1, 3),
                                  size=N_points), )
        elif plot_function == 'contour':
            X, Y = np.meshgrid(np.linspace(0, N_datasets*5, N_points),
                               np.linspace(0, N_datasets*5, N_points))
            Z = bivariate_normal(X, Y,
                                 np.random.uniform(1, 3),
                                 np.random.uniform(1, 3),
                                 np.random.uniform(0, N_datasets*5),
                                 np.random.uniform(0, N_datasets*5))
            d = (X,Y,Z)
        data.append(d)

    kwargs = dict(alpha=np.random.uniform(0.5,1.))
    def make_plot(style):
        with rc_context(style):
            rcParams['figure.dpi'] = 75
            rcParams['figure.facecolor'] = '#ffffff'

            fig,ax = plt.subplots(1,1,figsize=(6,4))
            for d in data:
                getattr(ax, plot_function)(*d, **kwargs)
            ax.set_xlabel('The x axis')
            ax.set_ylabel('The y axis')
            fig.set_facecolor('w')
            result = mpl_figure_data(fig)
            del fig
        return result

    return make_plot

def serve_page():
    s1, s2 = get_styles()
    assert s1 != s2

    plot_func = plot_generator()
    d1, d2 =  map(plot_func, (s1, s2))

    return render_template('main.html', image_1=d1, image_2=d2,
                           style_1=json.dumps(s1), style_2=json.dumps(s2))


def save_vote(win, lose):
    print('%s beats %s' % (win, lose))

@app.route('/vote/<int:winner>', methods=['POST'])
def vote(winner):
    data = request.values
    left = data['left']
    right = data['right']

    if winner == 0:
        win, lose = left, right
    else:
        win, lose = right, left
    save_vote(win, lose)

    return redirect('/')


@app.route("/")
def main():
    return serve_page()


if __name__ == "__main__":
    app.run(debug=True)
