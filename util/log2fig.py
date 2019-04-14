import matplotlib.pyplot as plt;
import sys;
import json;
import numpy as np;
logpath = sys.argv[1];
cd_lst = [];
cycle_lst = [];
cnt = 0;
dojob = False;
cycle = False;
with open(logpath,'r') as logfile:
    for line in logfile.readlines():
        if line.startswith("options:"):
            st = json.loads(line.lstrip("options:"));
            if (("girl" in  st["data"]) or ("bunny" in st["data"])) and (st["w"] == 0 or st["w"] ==0.25):
                if not st["grid_ply"]:
                    dojob = True;
                    cnt += 1;
                if st['w'] == 0.0:
                    cycle = False;
                if st['w'] > 0.0:
                    cycle =True;
        if line.startswith("#niter") and dojob:
            res = line.split(" ");
            cd_lst.append(float(res[1].lstrip("cd:")));
            cycle_lst.append(float(res[2].lstrip("inv:")));
            if line.startswith("#niter:4095"):
                dojob = False;
                plt.subplot(2,2,cnt);
                plt.plot(range(0,4096),np.log10(np.array(cd_lst)));
                if cycle:
                    plt.plot(range(0,4096),np.log10(np.array(cycle_lst)));
                cd_lst = [];
                cycle_lst = [];
    print(cnt);
plt.show()