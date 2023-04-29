#!python3

import subprocess
import itertools

def lsd_call(values):
    s, c, q, a, e, d, b = values
    cmd = ["./lsd"]

    cmd += ["-S", f"images/all_possibility/out_s_{s}_c_{c}_q_{q}_a_{a}_e_{e}_d_{d}_b_{b}.svg"]
    
    cmd += ["-s", f"{s}"]
    cmd += ["-c", f"{c}"]
    cmd += ["-q", f"{q}"]
    cmd += ["-a", f"{a}"]
    cmd += ["-e", f"{e}"]
    cmd += ["-d", f"{d}"]
    cmd += ["-b", f"{b}"]
    
    cmd += ["images/carte_XX_full.pgm", "out.txt"]

    process = subprocess.Popen(cmd)

    

s_values = [0.8]
c_values = [0.6]
q_values = [2]
a_values = [22.5]
e_values = [10]
d_values = [0.3]
b_values = [256]

import time

if __name__ == '__main__':
    res = list(itertools.product(s_values, c_values, q_values, a_values, e_values, d_values, b_values,))
    
    c = 0
    k = 1
    t0 = time.time()
    for args in res:
        lsd_call(args)
        
        c += 1
        if c == k:
            t = time.time()
            print(f"time for {c} execution : {t - t0}")
            k *= 2