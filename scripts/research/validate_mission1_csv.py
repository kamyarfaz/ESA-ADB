"""Validate an ESA Mission1 84-month prepared CSV without changing it."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--split',choices=['train','test'],required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    first=last=None;rows=0;labels_count={}
    columns=None
    for frame in pd.read_csv(args.path,chunksize=100000,low_memory=False):
        if columns is None:columns=list(frame.columns)
        if len(columns)!=175 or list(frame.columns)!=columns:
            raise ValueError('Expected a consistent 175-column Mission1 schema')
        timestamps=pd.to_datetime(frame.pop('timestamp'),format='%Y-%m-%d %H:%M:%S',errors='raise')
        if timestamps.isna().any():raise ValueError('Missing timestamp')
        if first is None:first=timestamps.iloc[0]
        if last is not None and timestamps.iloc[0]-last!=pd.Timedelta(seconds=30):raise ValueError('Gap between chunks')
        if not (timestamps.diff().iloc[1:]==pd.Timedelta(seconds=30)).all():raise ValueError('Non-30-second grid')
        last=timestamps.iloc[-1]
        label_columns=[c for c in frame if c.startswith('is_anomaly_')]
        features=[c for c in frame if c not in label_columns]
        if len(features)!=87 or {f'is_anomaly_{f}' for f in features}!=set(label_columns):raise ValueError('Feature/label schema mismatch')
        if not np.isfinite(frame[features].to_numpy(dtype=np.float64)).all():raise ValueError('Non-finite features')
        labels=frame[label_columns].to_numpy(dtype=np.float64)
        if not np.isin(labels,[0,1,2,3]).all():raise ValueError('Unexpected annotation codes')
        for label,count in zip(*np.unique(labels,return_counts=True)):labels_count[int(label)]=labels_count.get(int(label),0)+int(count)
        rows+=len(frame)
        if rows%1000000==0:print('Validated',rows,'rows',flush=True)
    start,end=('2000-01-01','2007-01-01') if args.split=='train' else ('2007-01-01','2014-01-01')
    if rows!=7364161 or first!=pd.Timestamp(start) or last!=pd.Timestamp(end):raise ValueError('Unexpected dataset extent')
    digest=hashlib.sha256()
    with args.path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    report={'path':str(args.path.resolve()),'split':args.split,'sha256':digest.hexdigest(),
            'rows':rows,'columns':len(columns),'start':str(first),'end':str(last),'label_counts':labels_count,
            'checks':'strict UTF-8, finite numeric features, paired labels, valid codes, complete 30-second timestamps and expected extent'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
