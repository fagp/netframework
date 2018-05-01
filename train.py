import time

from netutil.ExampleNet import ExampleNet

def main():
    # Common params
    sortingDL = ExampleNet('defaults')
    start=time.time()
    sortingDL.do_train()
    print('Total Training Time {:.3f}'.format(time.time()-start))

if __name__ == '__main__':
    main()
