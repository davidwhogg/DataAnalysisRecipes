import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','size':12})
rc('text', usetex=True)
rc('text.latex', preamble=open("hogg_style.tex").read())
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrow as Arrow
from matplotlib.patches import Rectangle as Rectangle
import numpy as np

class PGM(object):

    def __init__(self):
        self._nodes = {}
        self._edges = []
        self._plates = []
        return None

    def add_node(self, node):
        self._nodes[node.name] = node
        return None

    def add_edge(self, name1, name2, **kwargs):
        self._edges.append(Edge(self._nodes[name1], self._nodes[name2], **kwargs))
        return None

    def add_plate(self, plate):
        self._plates.append(plate)
        return None

    def render(self):
        fx, fy = 6.5, 6.5
        fig = plt.figure(figsize=(fx, fy))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False, xticks=[], yticks=[])
        ax.set_xlim(-0.5 * fx * 2.54, 0.5 * fx * 2.54)
        ax.set_ylim(-0.5 * fy * 2.54, 0.5 * fy * 2.54)
        for plate in self._plates:
            plate.render(ax)
        for edge in self._edges:
            edge.render(ax)
        for name, node in self._nodes.iteritems():
            node.render(ax)
        return ax

class Node(object):

    def __init__(self, name, content, x, y, diameter=1):
        self.name = name
        self.content = content
        self.x = x
        self.y = y
        self.diameter = diameter
        return None

    def render(self, ax):
        el = Ellipse(xy = [self.x, self.y],
                     width=self.diameter, height=self.diameter,
                     fc="none", ec="k")
        ax.add_artist(el)
        ax.text(self.x, self.y, self.content, ha="center", va="center")

class Edge(object):

    def __init__(self, node1, node2, **arrow_params):
        self.node1 = node1
        self.node2 = node2
        self.arrow_params = arrow_params
        return None

    def render(self, ax):
        x1, y1 = self.node1.x, self.node1.y
        x2, y2 = self.node2.x, self.node2.y
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx * dx + dy * dy)
        alpha1 = 0.5 * self.node1.diameter / dist
        alpha2 = 0.5 * self.node2.diameter / dist
        adx = dx * (1. - alpha1 - alpha2)
        ady = dy * (1. - alpha1 - alpha2)
        alen = np.sqrt(adx * adx + ady * ady)
        p = self.arrow_params
        p["ec"] = p.get("ec", "k")
        p["fc"] = p.get("fc", "k")
        p["head_length"] = p.get("head_length", 0.25)
        p["head_width"] = p.get("head_width", 0.1)
        ar = Arrow(x1 + alpha1 * dx, y1 + alpha1 * dy, adx, ady,
                   length_includes_head=True, width=0.,
                   **self.arrow_params)
        ax.add_artist(ar)
        return None

class Plate(object):

    def __init__(self, label, rect, **rect_params):
        self.label = label
        self.rect = rect
        self.rect_params = rect_params
        return None

    def render(self, ax):
        p = self.rect_params
        p["ec"] = p.get("ec", "k")
        p["fc"] = p.get("fc", "none")
        re = Rectangle([self.rect[0], self.rect[1]], self.rect[2], self.rect[3], **self.rect_params)
        ax.add_artist(re)
        # ax.text(self.rect[0], self.rect[1], self.label)
        ax.annotate(self.label, self.rect[:2], xycoords="data",
                    xytext=[5, 5], textcoords="offset points")
        return None

if __name__ == "__main__":
    thispgm = PGM()
    thispgm.add_node(Node("bar", r"$\alpha$", 2., 3.5))
    thispgm.add_node(Node("foo", r"$\allS$", 0., 0., diameter=1.5))
    thispgm.add_node(Node("baz", r"$\beta$", 0., 2., diameter=1.3))
    thispgm.add_edge("bar", "foo")
    thispgm.add_edge("baz", "foo")
    thispgm.add_plate(Plate(r"stars $k$", (-3., -3., 6., 4.)))
    thispgm.render().figure.savefig("test_pgm.pdf")
