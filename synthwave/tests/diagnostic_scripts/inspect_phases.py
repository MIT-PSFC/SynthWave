from diagnostic_header import np, xr

# Path to the data
ds_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/174956.nc"
ds = xr.open_dataset(ds_path)

# Parameters
t_val = 2.5
f_val = 10000  # 10 kHz
RAD2DEG = 180 / np.pi
tol = 10 / RAD2DEG

# Selection
t_idx = np.argmin(np.abs(ds.time.values - t_val))
f_idx = np.argmin(np.abs(ds.frequency.values - f_val))

# Extract data
real = ds.mirnov_fft_real.isel(idx=t_idx, frequency_idx=f_idx).values
imag = ds.mirnov_fft_imag.isel(idx=t_idx, frequency_idx=f_idx).values
signal = real + 1j * imag
phases = np.angle(signal)

# Coordinates
phis = np.radians(ds.sensor_phi.values)
thetas = np.arctan2(
    ds.sensor_Z.values, ds.sensor_R.values - ds.rmaxis.isel(idx=t_idx).values
)
names = ds.sensor_name.values

# Sorting
sorted_idx = np.argsort(thetas)
s_thetas = thetas[sorted_idx]
s_phis = phis[sorted_idx]
s_phases = phases[sorted_idx]
s_names = names[sorted_idx]

# Clustering
bands = []
if len(s_thetas) > 0:
    current_band = [0]
    for i in range(1, len(s_thetas)):
        if s_thetas[i] - s_thetas[current_band[0]] < tol:
            current_band.append(i)
        else:
            bands.append(current_band)
            current_band = [i]
    bands.append(current_band)

print(f"Number of bands found: {len(bands)}")
for b_idx, band in enumerate(bands):
    if len(band) < 2:
        continue

    ref_idx = band[0]
    ref_phi = s_phis[ref_idx]
    ref_phase = s_phases[ref_idx]

    print(
        f"\nBand {b_idx} (theta ~ {np.degrees(s_thetas[ref_idx]):.1f} deg), sensors={len(band)}"
    )

    for idx in band:
        rel_phi = (s_phis[idx] - ref_phi + np.pi) % (2 * np.pi) - np.pi
        rel_phase = (s_phases[idx] - ref_phase + np.pi) % (2 * np.pi) - np.pi

        # Calculate eff_n if rel_phi is large enough
        eff_n = rel_phase / rel_phi if abs(rel_phi) > 0.01 else 0

        print(
            f"  {s_names[idx]}: rel_phi={np.degrees(rel_phi):.1f}, rel_phase={np.degrees(rel_phase):.1f}, eff_n={eff_n:.2f}"
        )
