# NeAT

This repository contains the code for Neural fields for Adaptive Optical Two-photon Fluorescence Microscopy (NeAT). NeAT operates in three stages:

1. **Aberration and structural estimation**  
   From a single 3D image stack, NeAT estimates system or sample aberrations and recovers the underlying structural information, without any external training data.

2. **Conjugation error correction**  
   NeAT estimates and corrects conjugation errors in the imaging system, typically caused by incomplete conjugation or alignment errors.

3. **Sample motion correction**  
   NeAT maintains performance even with sample motion during the acquisition of the 3D input stack, by adaptively registering and correcting slice-to-slice motion artifacts.

---

### Code overview

This capsule includes three main scripts:

#### `neat_learning.py`  
Core routine that takes a 3D image stack as input and outputs aberration and structural estimations. Key physical parameters can be set via command-line arguments (a full list of tunable options is in the `args` parser):

- `psf_dx, psf_dy, psf_dz`: pixel sizes (µm) along $x$, $y$, $z$ axes  
- `cnts`: center coordinates of the input stack  
- `dims`: dimensions of the input stack  
- `na_exc`: excitation numerical aperture (NA)  
- `sample_motion`: Boolean parameter for sample motion correction. True if sample motion correction is performed.

**Expected outputs:**  
- **`rec.h5`** — HDF5 file containing:  
  - `out_x_m`: estimated structure  
  - `out_k_m`: estimated PSF  
  - `out_y`: computed image stack  
  - `wf`: Zernike coefficients  
  - `loss_list`: losses per epoch  
  - `y`: normalized input stack  
  - `y_min`, `y_max`: normalization bounds  
- **`est_aber_map.bmp`** — bitmap of the estimated aberration map from `wf` (unit: waves)
- **`slm_pattern.bmp`** — 8-bit grayscale image (with a pixel value of 255 corresponding to 1 wave) that is the wrapped corrective phase pattern to be applied to an SLM for aberration correction, if needed.

Feel free to explore the code and adjust parameters to suit your imaging setup.

#### `neat_conj_est.py`

This script estimates conjugation errors that may be present in a two-photon imaging system. It should be run after `neat_learning.py` on six image stacks. For testing, example stacks acquired using a microscope at UC Berkeley are available under `/commercial_beads_zeros/`, `/commercial_beads_mode4/`, `/commercial_beads_mode6/`, … and `/commercial_beads_mode13/`. Users should acquire these image stacks using their own system to estimate conjugation errors specific to their microscope. To acquire these stacks, users need to display on the SLM images corresponding to a flat phase pattern and five different Zernike modes for calibration (modes 4, 6, ..., 13), respectively. These patterns are available in `/slm_patterns_for_calibration/` as `k_vis_zeros.bmp`, `k_vis_mode4.bmp`, `k_vis_mode6.bmp`, ..., `k_vis_mode13.bmp.` (To apply these patterns directly on a SLM, the SLM needs to be calibrated so that a pixel value of 255 corresponds to 1 wave. The patterns can also be scaled and formatted for a deformable mirror.) Conjugation errors are computed as an affine transformation $\hat{H}$, from the reconstructions of these stacks and saved to `H.h5`.

#### `neat_conj_corr.py`

This script compensates for the conjugation errors estimated by `neat_conj_est.py` by applying the inverse affine transformation $\hat{H}^{-1}$ to an SLM pattern (generated as `slm_pattern.bmp` by `neat_learning.py`, which does not include conjugation error correction). The corrected pattern is saved as `k_vis_(args.dataset)_with_H.bmp`.

> **Note:** If you are using a microscope with perfect conjugation and alignment, you can skip `neat_conj_est.py` and `neat_conj_corr.py`.

---

### Running the Capsule

This capsule demonstrates NeAT’s application on both custom-built and commercial microscopes:

1. **Microscope with perfect conjugation and alignment**  
   Run NeAT on a fixed mouse-brain-slice image stack (data in `/custom_built_brain_slice/`), where conjugation-error correction is not required.

2. **Calibration for a microscope with possible conjugation and alignment issues (e.g., most commercial systems)**  
   Process five calibration stacks, estimate conjugation errors, and apply those corrections to generate an error-corrected SLM pattern.

3. **In vivo imaging with motion**  
   Run NeAT on an in vivo mouse-brain image stack with sample motion. If the samples do not require motion correction (e.g., zebrafish larvaes or plants) but still need conjugation error correction, simply set `sample_motion = False` and proceed with this step.
