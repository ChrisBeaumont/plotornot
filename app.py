# coding: utf-8

""" Plot or Not! """

from __future__ import division, print_function

# Standard library
import os, sys
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

# 'axes.color_cycle' : [],
randomize_rc = {'lines.marker' : Line2D.markers.keys(), 
                'lines.markeredgewidth' : np.arange(0,3,dtype=int),
                'lines.markersize' : np.arange(1,5,dtype=int)*10,
                'lines.linewidth' : np.arange(1,5,dtype=int),
                'lines.linestyle' : Line2D.lineStyles.keys(),
                'axes.edgecolor' : [(0.,0.,0.,x) for x in np.random.uniform(0., 0.5, size=10)],
                'axes.facecolor' : [(0.,0.,0.,x) for x in np.random.uniform(0.5, 1.0, size=10)],
                'axes.linewidth' : np.arange(1,5,dtype=int)}

def random_rc_pair(rcParams):
    """ Given a full rcParams dict, return two copies with one of the above 
        keys changed. 
    """
    rc1 = rcParams.copy()
    rc2 = rcParams.copy()
    
    param_to_change = random.choice(randomize_rc.keys())
    param_list = randomize_rc[param_to_change]
    random.shuffle(param_list)
    
    p1,p2 = param_list[:2]
        
    rc1[param_to_change] = p1
    rc2[param_to_change] = p2
    
    return rc1, rc2
    
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
            A = np.random.uniform(0., 5)
            B = np.random.uniform(0., 5)
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
@app.route("/")
def main():
    #s1, s2 = get_styles()
    #assert s1 != s2
    
    s1, s2 = random_rc_pair(rcParamsDefault)
    
    plot_func = plot_generator()
    d1, d2 =  map(plot_func, (s1, s2))
    return render_template('main.html', image_1=d1, image_2=d2)


if __name__ == "__main__":
    app.run(debug=True)
