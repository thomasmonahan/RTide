extract_tidegauge_from_ssh.ipynb

Load in slices of hourly AMM15_CHAMFER ssh data (per year).
Load in the combined 1993-2024 BODC 15min data, tg
Strip out the hourly instantaneous values from tg to match NEMO.

Manipulate, if necessary, tg for obs operator

Extract NEMO points at the tg locations.
Save as a tidegauge object with model and observation as functions of id_dim and t_dim

jelt
9 Apr 2025
sci-vm-01.jasmin

env: CANARI

attr_dict = {"year":str(year), \
             "originator": "jelt@noc.ac.uk", \
             "organisation": 'national_oceanography_centre,_liverpool',\
             "creation date": today_str, \
             "locations" : str(tg.dataset.sizes["id_dim"])+" UK Tidegauge network locations", \
             'obs_datum_information' : 'the_data_refer_to_admiralty_chart_datum_(acd)', \
             "obs_source": "compiled BODC observations 15min. Primary data with QC", \
             "model_source": "compiled from hourly AMM15_CHAMFER", \
             "qc_flags" : "(from BODC) N:nan, M:improbably, T:interpolated, no flag:OK", \
             "obs_script" : "to generate the obs data: bodc_qc_as_coast_obj.ipynb", \
             "model_obs_merge_script" : "extract_tidegauge_from_ssh.ipynb", \
             "Note1": "Merged AMM15_CHAMFER hourly SSH with thinned BODC data, using nearest model locations and matching timestamps", \



## Example usage
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

year = 2012
ofile = "/home/users/jelt/SURGE/AMM15_CHAMFER_gauge/AMM15_CHAMFER_1hr_BODC_"+str(year)+".nc"
ds=xr.open_dataset(ofile)

# Apply the QC mask without nuance
masked = ds \
    .where(ds.qc_flags != "N", np.NaN) \
    .where(ds.qc_flags != "M", np.NaN) \
    .where(ds.qc_flags != "T", np.NaN) 

delta_ssh = masked.model - masked.obs
delta_ssh_demeaned = delta_ssh - delta_ssh.mean(dim="t_dim")


# Plot and save the figure for the flags for all stations, per year
plt.close('all')
plt.figure(figsize=(10, 6))
plt.pcolormesh(ds.time, ds.site_name, delta_ssh_demeaned, shading='auto')  # Use shading='auto' for flexibility
plt.xticks(rotation=45)
plt.ylabel('station')
plt.xlabel(str(year))
plt.title('total water level difference: model - obs, local bias removed')
cbar = plt.colorbar(label="metres")

# Optionally, format x-axis ticks as dates if time is in datetime format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

             }
