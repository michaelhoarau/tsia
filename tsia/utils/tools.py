import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

def hex_to_rgb(hex_color):
    """
    Converts a color string in hexadecimal format to RGB format.
    
    PARAMS
    ======
        hex_color: string
            A string describing the color to convert from hexadecimal. It can
            include the leading # character or not
    
    RETURNS
    =======
        rgb_color: tuple
            Each color component of the returned tuple will be a float value
            between 0.0 and 1.0
    """
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], base=16) / 255.0 for i in [0, 2, 4])
    return rgb_color
    
def load_timeseries_from_dir(root_dir, time_extent=None, resampling='1H', clean_signals=True):
    """
    """
    list_files = os.listdir(root_dir)
    signal_list = []
    df_list = []
    
    progress_bar = tqdm(list_files)
    for f in progress_bar:
        progress_bar.set_description(f)
        progress_bar.refresh()
        
        fname = os.path.join(root_dir, f)
        signal_table = pq.read_table(fname)
        tag_df = signal_table.to_pandas()
        
        if time_extent is not None:
            start, end = time_extent
            tag_df = tag_df[start:end]
        
        if resampling is not None:
            tag_df = tag_df.resample(resampling).mean()

        # Clean the current dataframe and check if it's constant:
        is_constant = False
        if clean_signals == True:
            tag_df.replace(np.nan, 0.0, inplace=True)
            s = tag_df.describe().T
            is_constant = s[(s['max'] - s['min']) == 0].shape[0] > 0

        if is_constant == False:
            df_list.append(tag_df)
            signal_list.append(f.split('.')[0])
        
    return df_list, signal_list