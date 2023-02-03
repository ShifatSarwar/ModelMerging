from models import *
import time
from memory_profiler import profile


def addLine(name, line):
    with open(name, 'a') as f:
       f.write(line)
       f.write("\n")
    f.close()

@profile
def runAlgorithm(val, start_time):
    hist = getHistory(val, start_time)
    return hist


if __name__ == '__main__':
    val = 'tenSmall'
    start_time = time.time()
    # hist, yL, yP = runAlgorithm()  
    hist = runAlgorithm(val, start_time) 
    # plot_confusion(yL,yP, val)
    # plot_accuracies(hist, val)
    # plot_losses(hist, val)
    # plot_lrs(hist,val)
    timeTaken = time.time()-start_time
    dataLine = val+','+'2,'+str(timeTaken)
    addLine('dataList/list.csv',dataLine)
    print(val)
