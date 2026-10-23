"""Run an exported trusted SolarScope model against CSV weather."""
import argparse
import pandas as pd
from research import clean_history, forecast
from model_io import load_bundle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='Folder containing an extracted trusted model archive')
    parser.add_argument('--history',required=True)
    parser.add_argument('--weather',required=True)
    parser.add_argument('--output',default='forecast.csv')
    args = parser.parse_args()
    history,_ = clean_history(pd.read_csv(args.history))
    result = forecast(load_bundle(args.model),history,pd.read_csv(args.weather))
    result.to_csv(args.output,index=False)
    print(f'Saved {len(result)} forecast hours to {args.output}')
