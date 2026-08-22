# INSAT-3R Data Folder

Place your INSAT-3R satellite observation files (`.h5`) in this directory.

### Expected File Naming Format:
- Example: `3RIMG_19APR2026_0615_L1C_SGP_V01R00.h5`

### Expected Datasets inside each HDF5 file:
- `IMG_TIR1`: Thermal Infrared 1 Count Matrix `(1, 3207, 3062)`
- `IMG_TIR2`: Thermal Infrared 2 Count Matrix `(1, 3207, 3062)`
- `IMG_WV`: Water Vapor Count Matrix `(1, 3207, 3062)`
- `IMG_MIR`: Mid-Wave Infrared Count Matrix `(1, 3207, 3062)`
- `IMG_TIR1_TEMP`, `IMG_TIR2_TEMP`, `IMG_WV_TEMP`, `IMG_MIR_TEMP`: Kelvin Brightness Temperature Lookup Vectors `(1024,)`
- `X`, `Y`: Geostationary Coordinate Vectors

*Note: Large `.h5` satellite files are excluded from Git version control via `.gitignore`.*
