import multiprocessing as mp
import datetime
import math



def multicalculation(function, div):
    start_t = datetime.datetime.now()

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    param_dict = {'task1': list(range(0, div)),
                  'task2': list(range(div+1, div*2)),
                  'task3': list(range(div*2+1, div*3)),
                  'task4': list(range(div*3, div*4)),
                  'task5': list(range(div*4+1, div*5)),
                  'task6': list(range(div*5+1, div*6)),
                  'task7': list(range(div*6+1, div*7)),
                  'task8': list(range(div*7+1, div*8))}
    results = [pool.apply_async(function, args=param) for param in param_dict.items()]
    results = [p.get() for p in results]

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")