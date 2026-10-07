"""Render computed Figure 2 summaries with Schaefer ROI glass-brain insets.

The atlas is used only for visualization, never to re-parcellate response data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
from nilearn.plotting import plot_glass_brain
import numpy as np

from reproduce_figure2 import ROOT, MODELS, MODEL_NAMES, ROI_NAMES, DISPLAY_IDS, validate_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--mapping', type=Path, default=ROOT/'Parcel2ROI_new.npy')
    parser.add_argument('--atlas', type=Path, default=ROOT/'atlas/Schaefer1000_supplied_grid.nii.gz')
    args = parser.parse_args()
    with (args.results/'figure2_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    mapping = validate_mapping(args.mapping)
    atlas = nib.load(args.atlas)
    labels = atlas.get_fdata().astype(int)
    if not np.array_equal(np.unique(labels), np.arange(1001)):
        raise ValueError('Expected the complete 1000-parcel Schaefer atlas')
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes([.08, .27, .9, .49])
    colors = ['#000000', '#999999', '#ffa500', '#3c50ad', '#fa292e']
    for m, model in enumerate(MODELS):
        for j, roi_id in enumerate(DISPLAY_IDS):
            row, = [r for r in rows if r['model'] == model and r['roi'] == ROI_NAMES[roi_id-1]]
            low, high, median = [float(row[k]) for k in ['ci_low', 'ci_high', 'median']]
            x = j + (m-2)*.12
            ax.vlines(x, low, high, color=colors[m], lw=1)
            ax.plot(x, median, 'o', color=colors[m], ms=4.5,
                    label=MODEL_NAMES[m] if j == 0 else None)
    ax.set_xlim(-.5, 9.5)
    low = min(-5, min(float(r['ci_low']) for r in rows)-1)
    high = max(35, max(float(r['ci_high']) for r in rows)+1)
    ax.set_ylim(low, high)
    for x in np.arange(.5, 10, 1):
        ax.axvline(x, lw=.6, color='black', linestyle=':')
    ax.axhline(0, color='black', lw=.7)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, high, 5))
    ax.set_ylabel('Percent noise ceiling')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.03), frameon=False, fontsize=9)
    n, = {int(r['n']) for r in rows}
    fig.text(.08, .925, 'Figure 2 reproduction', fontsize=17)
    fig.text(.08, .885, f'I Knew You Were Black: N = {n}', fontsize=11, color='#555555')
    roi_metadata = []
    for j, roi_id in enumerate(DISPLAY_IDS):
        parcel_ids = np.flatnonzero(mapping == roi_id)+1
        mask = np.isin(labels, parcel_ids)
        inset = fig.add_axes([.08+j*.09, .14, .09, .12])
        display = plot_glass_brain(nib.Nifti1Image(mask.astype('float32'), atlas.affine),
                                  display_mode='l', axes=inset, figure=fig,
                                  colorbar=False, annotate=False, threshold=.5,
                                  cmap=ListedColormap(['black']), vmin=0, vmax=1,
                                  alpha=.25)
        fig.text(.125+j*.09, .12, ROI_NAMES[roi_id-1], ha='center', va='top', fontsize=9)
        xyz = nib.affines.apply_affine(atlas.affine, np.array(np.where(mask)).T)
        roi_metadata.append(dict(label=roi_id, name=ROI_NAMES[roi_id-1],
                                 parcel_ids_one_based=parcel_ids.tolist(),
                                 centroid_mni_mm=xyz.mean(0).tolist()))
    fig.text(.08, .035, 'Median across subjects; 95% confidence intervals from 10,000 bootstrap samples.',
             fontsize=8, color='#555555')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(args.results/f'figure2_with_rois.{ext}', dpi=220)
    plt.close(fig)
    metadata = dict(atlas='atlas/' + args.atlas.name, atlas_sha256=hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
                    source='https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI',
                    note='Supplied Schaefer 17-network parcellation, 2.5 mm grid, used for display only; supplied mapping retained.', rois=roi_metadata)
    (args.results/'roi_visualization_provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
