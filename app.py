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

app = Flask(__name__)


def get_styles():

    '''Return 2 random rcParams styles'''
    files = glob('params/*')
    random.shuffle(files)
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

def plot_generator():
    plot_function = random.choice(['plot', 'hist', 'scatter', 'contour'])

    N_points = 100
    N_datasets = np.random.randint(2, 8)
    data = []
    for ii in range(N_datasets):
        if plot_function in ['plot', 'scatter']:
            x = np.linspace(0., 10., N_points)
            y_func = random.choice([lambda x: 0.1*x, lambda x: 0.1*x**2, np.sin, np.cos])
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

    kwargs = dict(alpha=np.sqrt(np.random.random()))
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
    print '%s beats %s' % (win, lose)


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
