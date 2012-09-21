import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'size':12})
rc('text', usetex=True)
rc('text.latex', preamble=open("hogg_style.tex").read())
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.clf()
    plt.text(0, 0, r"$\allS$")
    plt.savefig("test_pgm.pdf")
    plt.savefig("test_pgm.png")
