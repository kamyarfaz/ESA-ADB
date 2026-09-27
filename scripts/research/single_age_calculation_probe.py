"""Standalone fixed arithmetic probe: no dataset reads or model code."""
import argparse,json,math,struct,sys,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    grid=1079295510000000000;native=1079295505179000000
    delta=grid-native;age=delta/1e9;value=math.log1p(age/30)
    expected32=struct.unpack('f',struct.pack('f',value))[0]
    began=time.monotonic();rows=[]
    for size,rounds in [(1,10000),(16,1000),(1024,100),(262144,20),(4469760,3)]:
        g=np.full(size,grid,dtype=np.int64);n=np.full(size,native,dtype=np.int64)
        counts={key:0 for key in ['delta','age','ratio','log_float64','log_float32','input_mutation']}
        examples=[]
        for trial in range(rounds):
            d=g-n;seconds=d/1e9;ratio=seconds/30;logged=np.log1p(ratio);actual32=logged.astype(np.float32)
            checks={'delta':d!=delta,'age':seconds!=age,'ratio':ratio!=(age/30),
                    'log_float64':np.abs(logged-value)>2*np.spacing(value),'log_float32':actual32!=expected32,
                    'input_mutation':(g!=grid)|(n!=native)}
            for stage,wrong in checks.items():
                count=int(np.count_nonzero(wrong));counts[stage]+=count
                if count and len(examples)<8:
                    index=int(np.flatnonzero(wrong)[0]);examples.append(dict(trial=trial,stage=stage,index=index,delta=int(d[index]),seconds=float(seconds[index]),log_float64=float(logged[index]),log_float32=float(actual32[index])))
        row=dict(size=size,rounds=rounds,counts=counts,examples=examples);rows.append(row);print(json.dumps(row),flush=True)
    # Independent scalar path in Python/math plus standard-library float32 packing.
    scalar_mismatches=0
    for _ in range(100000):
        answer=struct.unpack('f',struct.pack('f',math.log1p((grid-native)/1e9/30)))[0]
        scalar_mismatches+=answer!=expected32
    out=dict(python=sys.version,numpy=np.__version__,grid_ns=grid,native_ns=native,delta_ns=delta,seconds=age,
        reference_double=value,reference_float32=expected32,scalar_trials=100000,scalar_mismatches=scalar_mismatches,
        arrays=rows,seconds_elapsed=time.monotonic()-began,limitations='Bounded synthetic arithmetic test; a clean run does not explain or repair prior intermittent failures. Float64 log tolerance2ULP, float32 exact.')
    a.output.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
