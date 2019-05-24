import time
from netutil.loss_landscape import LandscapeVis

def main():
    errsurf = LandscapeVis('defaults')
    start=time.time()
    errsurf.do_plot()
    print('Total Training Time {:.3f}'.format(time.time()-start))

if __name__ == '__main__':
    main()
