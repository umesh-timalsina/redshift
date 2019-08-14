import os
import time

start_time = time.time()
ret = os.system('python3 swarp_wrapper_pool.py')
print('Time taken by multiprocessing implementation {:.4f}'.format(time.time()-start_time))

# start_time = time.time()
# ret = os.system('python3 prepare_dataset.py')
# print('Time taken by single process implementation {:.4f}'.format(time.time()-start_time))
