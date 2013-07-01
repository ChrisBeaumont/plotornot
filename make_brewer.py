"""
Make rcParams files for each of the qualitative color maps
"""

import brewer2mpl
import json
maps = 'Accent Dark2 Paired Pastel1 Pastel2 Set1 Set2 Set3'.split()

for m in maps:
    colors = brewer2mpl.get_map(m, 'qualitative', 5).mpl_colors
    data = {'axes.color_cycle': colors, 'lines.linewidth': 2.0}
    with open('params/'+ m + '.json', 'w') as outfile:
        outfile.write(json.dumps(data))
