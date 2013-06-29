# coding: utf-8

""" Plot or Not! """

from __future__ import division, print_function

# Standard library
import os
import sys
from cStringIO import StringIO
import json
import random
from glob import glob

# Third-party
from flask import Flask, request, render_template, redirect
from matplotlib import (rcParamsDefault, rc_context,
                        rcParams,
                        rc_params_from_file)
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
import numpy as np
import pymongo

app = Flask(__name__)

_all_colormaps = [m for m in plt.cm.datad]


def get_styles():
    '''Return 2 random rcParams styles'''
    files = glob('params/*')
    random.shuffle(files)

    styles = []
    for f in files[:2]:
        this_rc = rcParamsDefault.copy()
        if os.path.splitext(f)[1] == '.json':
            s = json.load(open(f))
        else:
            s = rc_params_from_file(f)

        for k, v in s.items():
            this_rc[k] = v
        styles.append(this_rc)
    return styles


def mpl_figure_data(f):
    """Convert a figure to a PNG data string"""
    data = StringIO()
    f.canvas.print_png(data)
    data = data.getvalue().encode('base64')
    return data


def plot_generator():
    plot_function = random.choice(['plot', 'hist', 'contourf'])

    N_points = 100
    N_datasets = np.random.randint(2, 8)
    data = []
    for ii in range(N_datasets):
        if plot_function == 'plot':
            x = np.linspace(0., 10., N_points)
            A = np.random.uniform(-5., 5)
            B = np.random.uniform(-5., 5)
            y_func = random.choice([lambda x: 0.1 * A * x + B,
                                    lambda x: 0.1 * A * x ** 2 + B,
                                    lambda x: A * np.sin(B * x),
                                    lambda x: A * np.cos(B * x)])
            y = y_func(x)
            d = (x, y)
        elif plot_function == 'hist':
            d = (np.random.normal(np.random.uniform(0, N_datasets * 5),
                                  np.random.uniform(1, 3),
                                  size=N_points), )
        elif plot_function == 'contourf':
            X, Y = np.meshgrid(np.linspace(0, N_datasets * 5, N_points),
                               np.linspace(0, N_datasets * 5, N_points))
            Z = bivariate_normal(X, Y,
                                 np.random.uniform(1, 3),
                                 np.random.uniform(1, 3),
                                 np.random.uniform(0, N_datasets * 5),
                                 np.random.uniform(0, N_datasets * 5))

            if len(data) == 0:
                d = [X, Y, Z]
            else:
                data[0][2] += Z
        data.append(d)

    kwargs = dict(alpha=np.random.uniform(0.5, 1.))

    def make_plot(style):
        cmap = style.pop('cmap', None)
        if cmap is not None:
            kwargs['cmap'] = cmap

        with rc_context(style):
            rcParams['figure.facecolor'] = '#ffffff'

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            for d in data:
                getattr(ax, plot_function)(*d, **kwargs)
            ax.set_xlabel('The x axis')
            ax.set_ylabel('The y axis')
            fig.set_facecolor('w')
            result = mpl_figure_data(fig)
            del fig
        return result

    return plot_function, make_plot


def serve_page():
    s1, s2 = get_styles()
    assert s1 != s2

    plot_type, plot_func = plot_generator()

    if plot_type == 'contourf':
        random.shuffle(_all_colormaps)
        cmap1, cmap2 = _all_colormaps[:2]
        s1['cmap'] = cmap1
        s2['cmap'] = cmap2

    d1, d2 = map(plot_func, (s1, s2))

    return render_template('main.html', image_1=d1, image_2=d2,
                           style_1=json.dumps(s1), style_2=json.dumps(s2),
                           plot_type=plot_type)


def save_vote(win, lose, plot_type=0, tie=False):
    uri = os.environ.get('MONGOLAB_URI', None)
    if uri is None:
        return

    if tie:
        post = {'win': win, 'lose': lose, 'plot_type': plot_type}
    else:
        post = {'tie_1': win, 'tie_2': lose, 'plot_type': plot_type}

    try:
        client = pymongo.MongoClient(uri)
        db = client.heroku_app16597650
        db.votes.insert(post)
    except pymongo.errors.OperationFailure:
        pass


@app.route('/vote/<int:winner>', methods=['POST'])
def vote(winner):
    data = request.values
    left = data['left']
    right = data['right']
    plot_type = data['plot_type']

    tie = False
    if winner == 0:
        win, lose = left, right
    elif winner == 1:
        win, lose = right, left
    else:
        win, lose = right, left
        tie = True

    save_vote(win, lose, plot_type, tie=tie)

    return redirect('/')


@app.route("/")
def main():
    return serve_page()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = '--debug' in sys.argv
    app.run(host='0.0.0.0', port=port, debug=debug)
