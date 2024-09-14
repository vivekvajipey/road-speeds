import os, argparse
from collections import defaultdict
import yaml
import math
import time
import json

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import cv2
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed


# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process satellite images and GPS tracks.")
    parser.add_argument('--in_tif_dirs', nargs='+', default=[
        'data/sat_tifs/sat_img', 'data/sat_tifs_2/batch2/R1_', 'data/sat_tifs_2/batch2/R2_', 'data/sat_tifs_2/batch2/R3_'])
    parser.add_argument('--in_gps_tracks', type=str, default='data/GPS_tracks.csv')
    parser.add_argument('--out_rgb_dir', type=str, default='data/rgb_crops_mo_4')
    parser.add_argument('--out_nir_dir', type=str, default='data/nir_crops_mo_4')
    parser.add_argument('--out_df_path', type=str, default='data/crop_info_base_4.csv')
    parser.add_argument('--valid_handles_yaml', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of jobs for parallel processing. Use -1 for all CPUs.")
    return parser.parse_args()

# Ensure that output directories exist
def ensure_directories_exist(*directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Read and process GPS tracks
def load_and_filter_gps_tracks(gps_tracks_path):
    gps_tracks = pd.read_csv(gps_tracks_path)
    gps_tracks = gps_tracks[~gps_tracks['speedKmHr'].isna()]
    gps_tracks = gps_tracks[gps_tracks['speedKmHr'] != 0]
    return gps_tracks

# Group GPS tracks by year and month
def group_gps_tracks_by_year_and_month(gps_tracks):
    month_dict = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    gps_tracks_year = {yr: gps_tracks[gps_tracks['date'].str[-4:] == yr] for yr in ('2021', '2022', '2023')}
    gps_tracks_year_month = {
        yr: {mo: gps_yr[gps_yr['date'].str[2:5] == mo] for mo in month_dict.values()}
        for yr, gps_yr in gps_tracks_year.items()
    }
    return gps_tracks_year_month, month_dict

# Get valid TIF file paths based on input directories and valid handles YAML (if any)
def get_tif_full_paths(in_tif_dirs, valid_handles_yaml=None):
    tif_full_paths = []
    if valid_handles_yaml is None:
        for dir in in_tif_dirs:
            for path in os.listdir(dir):
                full_path = os.path.join(dir, path)
                tif_full_paths.append(full_path)
    else:
        valid_handles = set(yaml.safe_load(open(valid_handles_yaml, 'r')))
        for dir in in_tif_dirs:
            for path in os.listdir(dir):
                handle = path[:-4]
                if handle in valid_handles:
                    full_path = os.path.join(dir, path)
                    tif_full_paths.append(full_path)
    return tif_full_paths


# Convert nested defaultdicts to regular dictionaries
def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def num_non_zero_neighbors(y, x, h, w, img, neigh_val=1):
    neighs = [(a, b) for a in range(y-1, y+2) for b in range(x-1, x+2)]
    ct = 0
    for neigh in neighs:
        if neigh[0] == 0 or neigh[0] == h-1 or neigh[1] == 0 or neigh[1] == w-1:
            continue
        if img[neigh] == neigh_val:
            ct += 1
    return ct

# Gets first non-zero neighbor
def get_non_zero_neighbor(y, x, h, w, img, neigh_val=1):
    neighs = [(a, b) for a in range(y-1, y+2) for b in range(x-1, x+2)]
    for neigh in neighs:
        if neigh[0] == 0 or neigh[0] == h-1 or neigh[1] == 0 or neigh[1] == w-1:
            continue
        if img[neigh] == neigh_val:
            return neigh
    return None


def read_image(ds):
    r_band = ds.GetRasterBand(3).ReadAsArray()
    g_band = ds.GetRasterBand(2).ReadAsArray()
    b_band = ds.GetRasterBand(1).ReadAsArray()
    nir_img = ds.GetRasterBand(4).ReadAsArray()
    rgb_img = np.dstack((r_band, g_band, b_band))
    np.nan_to_num(rgb_img, copy=False)
    rgb_img = (rgb_img / 10000 * 255).astype(np.uint8)
    np.nan_to_num(nir_img, copy=False)
    nir_img = (nir_img / 10000 * 255).astype(np.uint8)
    pixel_mask = np.where(nir_img > 0, 1, 0).astype(np.float16)
    return rgb_img, nir_img, pixel_mask


def job_task(tif_full_path, prev_lat_lons, img_ct_wrapper, img_id_map, gps_tracks_year_month, month_dict, args):
    tif_path = tif_full_path.split('/')[-1]
    img_path_base = tif_path[:-4]
    min_tracks = 1
    
    img_year = img_path_base.split('_')[-1][:4]
    img_month = month_dict[int(img_path_base.split('_')[-2])]
    # img_year_tracks = stgid_round_df[stgid_round_df['date'].str[-4:] == img_year]
    # img_month_tracks = img_year_tracks[img_year_tracks['date'].str[2:5] == img_month]
    img_month_tracks = gps_tracks_year_month[img_year][img_month]
    if img_month_tracks.shape[0] < min_tracks:
        print(f'Skipping {img_path_base} due to insufficient gps tracks ({img_month_tracks.shape[0]}, month).')
        return None

    img_gps_tracks = img_month_tracks

    ds = gdal.Open(tif_full_path)

    geotransform = ds.GetGeoTransform()
    full_x_min = geotransform[0]
    full_y_max = geotransform[3]
    full_x_max = full_x_min + geotransform[1] * ds.RasterXSize
    full_y_min = full_y_max + geotransform[5] * ds.RasterYSize

    for (lat_min, lon_min, lat_max, lon_max), (img_id, month_year_set) in prev_lat_lons.items():
        if math.isclose(full_y_min, lat_min) and math.isclose(full_x_min, lon_min) \
            and math.isclose(full_y_max, lat_max) and math.isclose(full_x_max, lon_max):
            if (img_month, img_year) in month_year_set:
                print(f'Skipping {img_path_base} due to repeat image.')
                return None
            else:
                month_year_set.add((img_month, img_year))
                true_image_id = img_id
                print(f'Found matching image {lat_min, lon_min, lat_max, lon_max} with id {true_image_id} and new month year {img_month, img_year}.')
                break
    else: # no break, so add new entry
        true_image_id = f'img{img_ct_wrapper[0]:04d}'
        prev_lat_lons[full_y_min, full_x_min, full_y_max, full_x_max] = [true_image_id, set([(img_month, img_year)])]
        img_ct_wrapper[0] += 1

    y_deg_t = lambda y: full_y_max + y * geotransform[5] # note geotransform[5] is neg
    x_deg_t = lambda x: full_x_min + x * geotransform[1]

    img_gps_tracks = img_gps_tracks[img_gps_tracks['latitude'] < full_y_max]
    img_gps_tracks = img_gps_tracks[img_gps_tracks['latitude'] > full_y_min]
    img_gps_tracks = img_gps_tracks[img_gps_tracks['longitude'] < full_x_max]
    img_gps_tracks = img_gps_tracks[img_gps_tracks['longitude'] > full_x_min]
    if img_gps_tracks.shape[0] < min_tracks:
        print(f'Skipping {img_path_base} due to insufficient gps tracks ({img_gps_tracks.shape[0]}, coords).')
        return None

    rgb_img, nir_img, pixel_mask = read_image(ds)
    road_path = skeletonize(pixel_mask)
    
    h, w = road_path.shape
    crop_sz = 224
    crop_every = 56 # pixels

    rgb_paths = []
    nir_paths = []
    lat_mins = [] 
    lat_maxs = []
    lon_mins = []
    lon_maxs = []
    speed_means = []
    speed_stds = []
    obs_cts = []

    crop_ct = 0
    while np.sum(road_path) > 0:
        path_len = 0
        
        # start at either leftmost or topmost point, depending on which has fewer neighbors
        ys, xs = np.nonzero(road_path)
        leftmost_idx = np.argmin(xs)
        leftmost_pt = (ys[leftmost_idx], xs[leftmost_idx])
        curr_pt = leftmost_pt
        left_num_neighs = num_non_zero_neighbors(leftmost_pt[0], leftmost_pt[1], h, w, road_path)
        if left_num_neighs > 1:
            topmost_idx = np.argmin(ys)
            topmost_pt = (ys[topmost_idx], xs[topmost_idx])
            top_num_neighs = num_non_zero_neighbors(topmost_pt[0], topmost_pt[1], h, w, road_path)
            if top_num_neighs < left_num_neighs:
                curr_pt = topmost_pt
        
        while True:
            road_path[curr_pt] = 0
            y_min, x_min, y_max, x_max = (int(curr_pt[0] - crop_sz/2), int(curr_pt[1] - crop_sz/2),
                                        int(curr_pt[0] + crop_sz/2), int(curr_pt[1] + crop_sz/2))
            if path_len % crop_every == 0 and y_min >= 0 and x_min >= 0 and y_max < h and x_max < w:
                y_deg_min, x_deg_min, y_deg_max, x_deg_max = (y_deg_t(y_max), x_deg_t(x_min), y_deg_t(y_min), x_deg_t(x_max))
                
                img_crop_gps = img_gps_tracks[img_gps_tracks['latitude'] < y_deg_max]
                img_crop_gps = img_crop_gps[img_crop_gps['latitude'] > y_deg_min]
                img_crop_gps = img_crop_gps[img_crop_gps['longitude'] < x_deg_max]
                img_crop_gps = img_crop_gps[img_crop_gps['longitude'] > x_deg_min]

                if len(img_crop_gps) > min_tracks: # do crop
                    rgb_paths.append(os.path.join(args.out_rgb_dir, f'{true_image_id}_{img_month}_{img_year}_rgb_{crop_ct}.png'))
                    nir_paths.append(os.path.join(args.out_nir_dir,f'{true_image_id}_{img_month}_{img_year}_nir_{crop_ct}.png'))
                    lat_mins.append(y_deg_min)
                    lat_maxs.append(y_deg_max)
                    lon_mins.append(x_deg_min)
                    lon_maxs.append(x_deg_max)
                    speed_means.append(img_crop_gps['speedKmHr'].mean())
                    speed_stds.append(img_crop_gps['speedKmHr'].std())
                    obs_cts.append(len(img_crop_gps))

                    rgb_crop = rgb_img[y_min:y_max, x_min:x_max]
                    Image.fromarray(rgb_crop).save(rgb_paths[-1], compression_level=0)
                    nir_crop = nir_img[y_min:y_max, x_min:x_max]
                    Image.fromarray(nir_crop).save(nir_paths[-1], compression_level=0)

                    crop_ct += 1
                
            curr_pt = get_non_zero_neighbor(curr_pt[0], curr_pt[1], h, w, road_path)
            if curr_pt is None:
                break
            path_len += 1
    img_df = pd.DataFrame({
        'rgb_path': rgb_paths,
        'nir_path': nir_paths,
        'lat_min': lat_mins,
        'lat_max': lat_maxs,
        'lon_min': lon_mins,
        'lon_max': lon_maxs,
        'speed_mean': speed_means,
        'speed_std': speed_stds,
        'num_obs': obs_cts,
        'img_id': [true_image_id for _ in range(len(obs_cts))],
    })
    print(f'!!! Generated {crop_ct} crops from satellite image {tif_path}')
    if crop_ct > 0:
        img_id_map[true_image_id][img_year][img_month] = tif_full_path
    # img_df.to_csv(f'~/tmp/{img_path_base}_{int(time.time())}.csv', index=False)
    return img_df


# Main function to handle the full process
def main():
    args = parse_arguments()

    # Ensure directories exist
    ensure_directories_exist(args.out_rgb_dir, args.out_nir_dir)

    # Load and process GPS tracks
    gps_tracks = load_and_filter_gps_tracks(args.in_gps_tracks)

    # Group GPS tracks by year and month
    gps_tracks_year_month, month_dict = group_gps_tracks_by_year_and_month(gps_tracks)

    # Get full TIF file paths
    tif_full_paths = get_tif_full_paths(args.in_tif_dirs, args.valid_handles_yaml)

    # Initialize necessary data structures for image processing
    prev_lat_lons = {}  # Tracks previous latitude/longitude boundaries
    img_id_map = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))  # Mapping for image IDs
    img_ct_wrapper = [0]  # Counter for image IDs

    # Process each TIF file, using parallel processing if n_jobs != 1
    print(f'Generating crops from {len(tif_full_paths)} inputs...')

    if args.n_jobs == 1:
        # Single process execution
        img_dfs = [job_task(path, prev_lat_lons, img_ct_wrapper, img_id_map, gps_tracks_year_month, month_dict, args) for path in tif_full_paths]
    else:
        # Parallel execution using joblib
        img_dfs = Parallel(n_jobs=args.n_jobs)(
            delayed(job_task)(path, prev_lat_lons, img_ct_wrapper, img_id_map, gps_tracks_year_month, month_dict, args)
            for path in tif_full_paths
        )

    img_dfs = [img_df for img_df in img_dfs if img_df is not None]

    # Combine all DataFrames into one and save results
    out_df = pd.concat(img_dfs, axis=0)
    out_df.to_csv(args.out_df_path, index=False)

    # Save image ID map to JSON
    img_id_map = default_to_regular(img_id_map)
    json.dump(img_id_map, open('data/img_id_map_4.json', 'w'), indent=2)



if __name__ == "__main__":
    main()
