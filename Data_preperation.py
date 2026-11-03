#Load Data from Database
import os
import pandas as pd
import sqlite3
import numpy as np
import zipfile
from datetime import date
import json
#import matplotlib.pyplot as plt


class ProfoundData:
    '''
    Extract necessary data from Profound DataBase and preprocess it
    '''
    def __init__(self, db_source): #handsoff not needed
        #self.split = split #possible splits later !

        if isinstance(db_source, list):
            db_source= db_source[0]

        if zipfile.is_zipfile(db_source):
            print('Zip file found')
            #unzip file
            with zipfile.ZipFile(db_source, mode='r') as zf:
                path = db_source.replace('.zip', '')  
                zf.extract('ProfoundData.sqlite', path=path)
                self.path_to_db = path + 'ProfoundData.sqlite'

        elif db_source.endswith('.sqlite'):
            print('Sqlite file found')
            self.path_to_db = db_source

        # elif zipfile.is_zipfile(db_source):
        #         #unzip file
        #         with zipfile.ZipFile(db_source, mode='r') as zf:
        #             path= db_source.strip('ProfoundData.zip')
        #             zf.extract('ProfoundData.sqlite', path=path)
        #             self.path_to_db = path + 'ProfoundData.sqlite'
        else: 
            raise Exception('Unsupported datatype or path provided. Expected a zipfile or a SQLite database file.')
        
        #connect to database
        self.con = sqlite3.connect(self.path_to_db)
        #self.handsoff = handsoff

    def get_table(self, table_name, connection):
        '''
        get table from database
        table_name (str): Name of table
        connection: established connection to database
        '''
        df = pd.read_sql_query(''.join(('SELECT * FROM ', table_name)), connection)
        return df

    def get_clim(self, station_id):
        '''
        get dataset climate: Precipitation & temperature
        station_id: int
        '''
        data = self.get_table('CLIMATE_LOCAL', self.con)                    
        data = data.loc[data['site_id'] == station_id, ['date', 'rad_Jcm2', 'p_mm', 'tmean_degC']] #Should work now, daily resolution, so no conversions there. If it cant find the index try rad_Jcm2day instead
        data = data.set_index(pd.to_datetime(data['date']))
        data['Precip'] = data['p_mm'].copy()
        data['Tair'] = data['tmean_degC'].copy()

        # reference: Taiz & Zeiger (2015): Plant Physiology and Development, Ch.9 (p.172-174)
        # total daily global radiation (J/cm²/day) into PPFD (µmol photons/m²/s)
        data['PAR'] = ((data.rad_Jcm2 * (2.2*(10**(-7))) / (299792458 * (6.626070150 * (10 **(-34)))))/(6.02*(10**23))) / 0.0001
        data = data.drop(['date', 'p_mm', 'tmean_degC', 'rad_Jcm2'], axis=1)
        return data

    def get_fapar(self, station_id, interp_max_gap_days=20):
        '''
        MODIS FPAR (%). Returns a daily series with single-day gaps
        filled by neighbor average; edge single gaps copy the neighbor.
        Longer gaps remain NaN.
        '''
        datafpar = self.get_table('MODIS', self.con)
        data = datafpar.loc[datafpar['site_id'] == station_id, ['date', 'fpar_percent']]
        data = data.set_index(pd.to_datetime(data['date']))
        data['fapar'] = data['fpar_percent'].copy()
        data = data.drop(['date', 'fpar_percent'], axis=1)
        start = data.index[0]
        end = data.index[-1]
        data = data[data.index.isin(pd.date_range(start, end))]['fapar']
        #cleaning data
        idx = data[np.isnan(data)].index
        # mean bevor and after
        for i in idx:
            after = data.shift(periods=-1) 
            bevor = data.shift(periods=1)
            if np.isfinite(bevor[i]) and np.isfinite(after[i]):
                data[i] = np.nanmean([bevor[i], after[i]]) #fill missing timestep with mean of one timestep before and after
            elif np.isfinite(bevor[i]):
                data[i] = bevor[i]
            elif np.isfinite(after[i]):
                data[i] = after[i]

        newdata = pd.DataFrame(data={'date': pd.date_range(start, end)})
        newdata = newdata.set_index(pd.to_datetime(newdata['date'])).drop(['date'], axis=1)
        fapar = newdata.join(data) #timestep is set
        nans = np.where(np.isnan(fapar.fapar))[0] 
        # set value for days of resolution (usually 16 following days)
        for i in nans:
            fapar.iloc[i]['fapar'] = fapar.iloc[i - 1]['fapar'] #replace data with data from timstep before, if there is nan in current timestep
        data = fapar

        return data #returned as a fraction
        
    def get_vpd(self, station_id):
        '''
        get dataset METEOROLOGICAL: Vapour pressure deficit
        '''
        datavpd = self.get_table('METEOROLOGICAL', self.con)
        data = datavpd.loc[datavpd['site_id'] == station_id, ['date', 'vpdFMDS_hPa']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1) #delete date in column to have time series
        data = data.resample('D').mean() # data is taken in 30min intervals, this converts it to a daily resolution
        data['VPD'] = data['vpdFMDS_hPa'].copy()/10 #hPa to kPa
        data = data.drop(['vpdFMDS_hPa'], axis=1)
        return data

    def get_et(self, station_id):
        '''
        get dataset ATMOSPHERICHEATCONDUCTION -  Latent heat flux
        requires: 
        '''
        dataet = self.get_table('ATMOSPHERICHEATCONDUCTION', self.con)
        data = dataet.loc[dataet['site_id'] == station_id, ['date', 'leCORR_Wm2']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1)                                      # delete date in column to have time series
        data = data.resample('D').mean()                                        # convert to daily resolution # check if already, or should be sum!!!!!
        data['LE'] = data['leCORR_Wm2'].copy()                                  # "W m-2; Latent heat flux, corrected le_FMDS by energy balance closure correction factor"
        data = data.drop(['leCORR_Wm2'], axis=1)                                # Rate: energy flux per second per m²
        
        # correct neg values
        pos = data.loc[data['LE'] < 0.0]
        bevor = data.shift(periods=1)
        after = data.shift(periods=-1)
        for p in pos.index:
            if bevor['LE'][p] >= 0 and after['LE'][p] >= 0:
                data['LE'][p] = np.mean([bevor['LE'][p], after['LE'][p]])
            elif bevor['LE'][p] >= 0:
                data['LE'][p] = bevor['LE'][p]
            elif after['LE'][p] >= 0:
                data['LE'][p] = after['LE'][p]
            else:
                data['LE'][p] = 0
        
        return data

    def get_gpp(self, station_id):
        '''
        get Flux dataset: Each value represents the average flux during that 30-minute period, in µmol CO₂ per m² per second
        '''
        datagpp = self.get_table('FLUX', self.con)
        data = datagpp.loc[datagpp['site_id'] == station_id, ['date', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1']]
        data = data.set_index(pd.to_datetime(data['date']))
        """
        formula converts the Gross Primary Productivity (GPP) values: µmol CO2/m²/s to g C/m²/day
        10 **-6 converts the values from micromoles (µmol) to moles (mol)
        2.011: This factor converts the amount of C in moles to grams
        1800: This converts the interval 30min to per day. 
        
        PRELES: Gross primary production (P) g C m–2
        """
        data['GPP'] = data['gppDtVutRef_umolCO2m2s1'].values.copy() * 10 ** -6 * 12.011 * 1800   # 1800 = 30min in seconds
       
        data = data.drop(['date', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1'], axis=1)
        data = data.resample('D').sum()                                     #sum every interval together for daily measurement.
        return data

    def get_swc(self, station_id):
        """
        Soil water content time series (daily).
        Returns columns:
        - SW            : measured (%)  – same as before (for backward compatibility)
        - SW_frac       : measured fraction (0..1)
        - SW_clim_pct   : DOY climatology median (%), built from all years
        - SW_filled_pct : SW with NaNs filled from SW_clim_pct
        
        """
        dataswc = self.get_table('SOILTS', self.con)
        
        # --- SOILTS: daily series ---
        data = dataswc.loc[dataswc['site_id'] == station_id, ['date', 'swcFMDS1_degC', 'swcFMDS1_qc']].copy()
        data = data.set_index(pd.to_datetime(data['date'])).drop(columns=['date'])
        if 'swcFMDS1_qc' in data:
            data.loc[~data['swcFMDS1_qc'].isin([0,1]), 'swcFMDS1_degC'] = np.nan
            data = data.drop(columns=['swcFMDS1_qc'])
        data = data.resample('D').mean()
        
        data['SW'] = data['swcFMDS1_degC'] # PROFOUND: volumetric %, see docs
        data = data.drop(columns=['swcFMDS1_degC'])
        
        return data #, nan_periods if 'nan_periods' in locals() else data
    
    @staticmethod
    def sw_percent_to_mm(sw_percent, soildepth_mm):
        return float(soildepth_mm * (sw_percent / 100.0))
    
    @staticmethod
    def clamp_mm(x, soildepth_mm, ThetaFC, ThetaPWP):
        lo = soildepth_mm * max(0.0, ThetaPWP)
        hi = soildepth_mm * max(0.0, ThetaFC)
        if hi <= lo: hi = lo + 1e-6
        return float(np.clip(x, lo, hi))
    
    @staticmethod
    def compute_swinit_prior_measured(sw_df, soildepth_mm, ThetaFC, ThetaPWP,
                                  start_date=None, window_days=15, min_obs=5):
        
        vals = sw_df['SW'] if 'SW' in sw_df.columns else sw_df['SW_perc']
        vals = vals.dropna()
        
        if start_date is not None and len(vals) >= min_obs:
            doy0 = pd.Timestamp(start_date).dayofyear
            d = np.minimum(np.abs(vals.index.dayofyear - doy0), 366 - np.abs(vals.index.dayofyear - doy0))
            sub = vals[d <= window_days]
            
            if len(sub) >= min_obs:
                v_mm = ProfoundData.sw_percent_to_mm(np.median(sub.values), soildepth_mm)
                return ProfoundData.clamp_mm(v_mm, soildepth_mm, ThetaFC, ThetaPWP), "seasonal_median_measured_pct"
            
        if len(vals) >= min_obs:
            v_mm = ProfoundData.sw_percent_to_mm(np.median(vals.values), soildepth_mm)
            return ProfoundData.clamp_mm(v_mm, soildepth_mm, ThetaFC, ThetaPWP), "overall_median_measured_pct"
        
        # fallback: midpoint prior
        mid = soildepth_mm * (ThetaPWP + 0.5*(ThetaFC - ThetaPWP))
        return ProfoundData.clamp_mm(mid, soildepth_mm, ThetaFC, ThetaPWP), "midpoint_between_PWP_and_FC"

    @staticmethod
    def sw_percent_df_to_mm(sw_df, soildepth_mm, ThetaFC=None, ThetaPWP=None, clamp=False, out_col="SW"):
        """
        Convert a SW percent DataFrame to mm of water over 'soildepth_mm'.
        sw_df: DataFrame with a column 'SW' or 'SW_perc' in percent (0..100).
        Returns a 1-col DataFrame with column name 'out_col' (mm).
        """
        import numpy as np
        import pandas as pd

        if sw_df is None or sw_df.empty:
            return pd.DataFrame(index=pd.Index([], name=getattr(sw_df.index, 'name', None)), columns=[out_col])

        col = 'SW' if 'SW' in sw_df.columns else ('SW_perc' if 'SW_perc' in sw_df.columns else None)
        if col is None:
            raise KeyError("sw_percent_df_to_mm: expected 'SW' or 'SW_perc' column in sw_df.")

        # Conversion: transfer volumetric soil moisture to mm
        mm = (sw_df[col].astype(float) / 100.0) * float(soildepth_mm)

        if clamp and ThetaFC is not None and ThetaPWP is not None:
            lo = float(soildepth_mm) * max(0.0, float(ThetaPWP))
            hi = float(soildepth_mm) * max(0.0, float(ThetaFC))
            if hi <= lo:
                hi = lo + 1e-6
            mm = np.clip(mm, lo, hi)

        out = mm.to_frame(name=out_col)
        return out


    # def shorten_merge(self, GPP, ET, Clim, VPD, fAPAR, SW): #
    #     out = GPP.merge(ET, how='inner', on=['date']).merge(Clim, how='inner', on=['date']).merge(VPD, how='inner', on=['date']).merge(fAPAR, how='inner', on=['date']).merge(SW, how='inner', on=['date'])
    #     return out
    
    def shorten_merge(self, *dfs):
        out = dfs[0]
        for d in dfs[1:]:
            out = out.join(d, how='inner')  # inner on index (dates)
        return out
    
    def availability_report_and_filter(op: pd.DataFrame,
                                   station_id: int,
                                   essential_vars=None,
                                   require_complete_years: bool = True,
                                   verbose: bool = True):
        """
        Build a data-availability report and (optionally) filter to periods with all essential vars.
        Returns:
        output            : filtered DataFrame (None if no qualifying period)
        report            : list of dicts [{start_date, end_date, missing_vars}, ...]
        adjusted_periods  : list of (start_ts, end_ts) covering only complete years
        mask              : boolean mask (same index as op) used to filter, or None
        Notes:
        - This expects 'op' to have a daily DatetimeIndex.
        - Default essential_vars excludes SW and uses LE (BEFORE you convert LE→ET).
        """
        # defaults (same as your original code)
        if essential_vars is None:
            essential_vars = ['GPP', 'Precip', 'Tair', 'PAR', 'VPD', 'fapar', 'LE']

        # ensure sorted by date
        op = op.sort_index()
        idx = op.index

        # sanity: only check columns that actually exist
        check_cols = [c for c in essential_vars if c in op.columns]
        if not check_cols:
            if verbose:
                print("No essential variables found in dataframe columns; nothing to check.")
            return op.copy(), [], [], None

        # work on a boolean frame of missingness
        md = op[check_cols].isna().copy()

        # group contiguous stretches with identical 'missing_vars'
        md['missing_vars'] = md.apply(lambda row: tuple([v for v in check_cols if row[v]]), axis=1)
        md['date'] = idx
        md['date_diff'] = md['date'].diff()
        md['missing_vars_shifted'] = md['missing_vars'].shift()
        md['new_group'] = (md['missing_vars'] != md['missing_vars_shifted']) | (md['date_diff'] != pd.Timedelta('1D'))
        md['group'] = md['new_group'].cumsum()

        report = []
        for _, grp in md.groupby('group'):
            start_date = grp['date'].iloc[0]
            end_date   = grp['date'].iloc[-1]
            miss       = grp['missing_vars'].iloc[0]
            report.append({'start_date': start_date, 'end_date': end_date, 'missing_vars': miss})

        if verbose:
            print(f"\nData availability report for site {station_id}:")
            for item in report:
                start = item['start_date'].date()
                end   = item['end_date'].date()
                miss  = item['missing_vars']
                if len(miss) == 0:
                    print(f"From {start} to {end}: All essential data available")
                else:
                    print(f"From {start} to {end}: Missing essential variables: {', '.join(miss)}")

        # periods where essentials are fully available
        available_periods = [(r['start_date'], r['end_date']) for r in report if len(r['missing_vars']) == 0]

        # adjust to complete years (optional; matches your original behavior)
        adjusted_periods = []
        if require_complete_years:
            for start_date, end_date in available_periods:
                # move start to next Jan 1 if needed
                adjusted_start = pd.Timestamp(f"{start_date.year + (start_date.month != 1 or start_date.day != 1):04d}-01-01")
                # move end to previous Dec 31 if needed
                adjusted_end   = pd.Timestamp(f"{end_date.year - (end_date.month != 12 or end_date.day != 31):04d}-12-31")
                if adjusted_start <= adjusted_end:
                    adjusted_periods.append((adjusted_start, adjusted_end))
        else:
            adjusted_periods = available_periods

        if not adjusted_periods:
            if verbose:
                print("No complete years with full essential data are available for calculations.")
            return None, report, [], None

        if verbose:
            print("\nAdjusted periods with complete years and essential data:")
            for s, e in adjusted_periods:
                print(f"From {s.date()} to {e.date()}")

        # build mask for the union of adjusted periods
        masks = [(op.index >= s) & (op.index <= e) for s, e in adjusted_periods]
        if not masks:
            if verbose:
                print("No data available after adjusting to complete years.")
            return None, report, adjusted_periods, None

        full_mask = np.logical_or.reduce(masks)

        # final filtered output: only rows inside periods and with no missing essentials
        output = op.loc[full_mask].dropna(subset=check_cols)

        return output, report, adjusted_periods, full_mask

    def estimate_fc_pwp_from_texture(sand, silt, clay):
        # simple heuristics (replace with a pedotransfer if you prefer)
        if clay is None and sand is None:
            return 0.30, 0.15  # generic loam-ish
        if clay is not None and clay >= 40:      # clayey
            return 0.42, 0.27
        if sand is not None and sand >= 65:      # sandy
            return 0.18, 0.08
        # loam / sandy-loam
        return 0.30, 0.15

    def get_soil_params(self, station_id):
        so = self.get_table('SOIL', self.con)
        so = so.loc[so['site_id'] == station_id].copy()
        ThetaFC = np.nan
        ThetaPWP = np.nan
        soildepth = np.nan

        def med(col):
            return float(np.nanmedian(so[col].astype(float))) if col in so.columns and so[col].notna().any() else np.nan

        fc_pct   = med('fcapv_percent')
        pwp_pct  = med('wiltpv_percent')   
        por_pct  = med('porosity_percent')
        whc_mm   = med('whc_mm')
        ld_cm    = np.nanmax(so['lowerDepth_cm'].astype(float)) if 'lowerDepth_cm' in so.columns and so['lowerDepth_cm'].notna().any() else np.nan
        sand_pct = med('sand_percent')
        silt_pct = med('silt_percent')
        clay_pct = med('clay_percent')
        

        # soildepth: deepest horizon lower bound if available
        if np.isfinite(ld_cm):
            soildepth = float(ld_cm) * 10.0
        else:
            soildepth = float(413)

        # Case A: both FC & PWP given
        if np.isfinite(fc_pct) and np.isfinite(pwp_pct):
            ThetaFC  = fc_pct / 100.0
            ThetaPWP = pwp_pct / 100.0

        # Case B: AWC + porosity/texture
        if not (np.isfinite(ThetaFC) and np.isfinite(ThetaPWP)):
            AWC = np.nan
            if np.isfinite(whc_mm) and np.isfinite(soildepth) and soildepth > 0:
                AWC = whc_mm / soildepth
            if (not np.isfinite(ThetaFC)) and np.isfinite(por_pct):
                ThetaFC = 0.9 * (por_pct / 100.0)  # FC < porosity
            if not np.isfinite(ThetaFC):
                ThetaFC, ThetaPWP_guess = ProfoundData.estimate_fc_pwp_from_texture(sand_pct, silt_pct, clay_pct) #Could change this to a formal pedotransfer function Saxton & Rawls (2006)
            if np.isfinite(AWC):
                # if PWP still unknown, derive from AWC
                if not np.isfinite(ThetaPWP):
                    ThetaPWP = max(0.05, ThetaFC - AWC)
            # final sanity if still missing PWP
            if not np.isfinite(ThetaPWP):
                _, ThetaPWP = ProfoundData.estimate_fc_pwp_from_texture(sand_pct, silt_pct, clay_pct)

        # sanity clamp
        if np.isfinite(ThetaFC) and np.isfinite(ThetaPWP) and ThetaPWP >= ThetaFC:
            ThetaPWP = max(0.05, ThetaFC - 0.02)
        if not np.isfinite(ThetaFC):
            ThetaFC = float(('ThetaFC', 0.45))
        if not np.isfinite(ThetaPWP):
            ThetaPWP = float(('ThetaPWP', 0.118))
        if not np.isfinite(soildepth):
            soildepth = float(('soildepth', 500.0))
        

        return ThetaFC, ThetaPWP, soildepth

    def get_site_meta(self, station_id: int):
    
        # --- STAND (LAI) --- #leaf Area Index ; helps with interception (CWmax)
        stand = self.get_table('STAND', self.con)
        st = stand.loc[stand['site_id'] == station_id]
        lai_med = np.nan
        if 'lai' in st.columns and not st.empty and st['lai'].notna().any():
            lai_med = float(np.nanmedian(st['lai'].astype(float)))  # LAI is adimensional
        
        if not np.isfinite(lai_med):
            df = self.get_table('MODIS', self.con)
            df = df.loc[df['site_id'] == station_id, ['date', 'lai']].copy()
            if not df.empty and df['lai'].notna().any():
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                # (optional) handle duplicate days
                if df.index.has_duplicates:
                    df = df.groupby(level=0).mean(numeric_only=True)
                lai_med = float(np.nanmedian(df['lai'].astype(float)))

            

        # --- TREE (dominant species) ---
        tree = self.get_table('TREE', self.con)
        tr = tree.loc[tree['site_id'] == station_id]
        species = (tr['species'].mode().iat[0] if 'species' in tr.columns and len(tr) else None)

        # classify canopy type roughly
        def _is_conifer(name):
            if not isinstance(name, str): return False
            name_l = name.lower()
            return any(k in name_l for k in ['pinus', 'picea', 'abies', 'larix']) #soro - 'Fagus sylvatica' -> deciduous

        is_conifer = _is_conifer(species)

        # --- SOIL (ThetaFC, ThetaPWP, soildepth) ---
        soil = self.get_table('SOIL', self.con)
        so = soil.loc[soil['site_id'] == station_id]
        
        ThetaFC, ThetaPWP, soildepth = self.get_soil_params(station_id)

        # ThetaFC = float(so['fcapv_percent'].median()/100.0) if 'fcapv_percent' in so else np.nan
        # ThetaPWP = float(so['wiltpv_percent'].median()/100.0) if 'wiltpv_percent' in so else np.nan

        # # soildepth: deepest horizon lower limit (cm → mm). If missing, keep NaN and let defaults stand.
        # if 'lowerDepth_cm' in so and len(so['lowerDepth_cm'].dropna()):
        #     soildepth = float(np.nanmax(so['lowerDepth_cm'])) * 10.0
        # else:
        #     soildepth = np.nan

        # sanity: keep ThetaPWP < ThetaFC
        if np.isfinite(ThetaFC) and np.isfinite(ThetaPWP) and ThetaPWP >= ThetaFC:
            # nudge slightly apart
            ThetaPWP = max(0.0, ThetaFC - 0.02)

        # --- CWmax from LAI rule-of-thumb ---
        if np.isfinite(lai_med):
            # ballpark: conifer ~0.7 mm per LAI, broadleaf ~0.4 mm per LAI
            cw_coef = 0.7 if is_conifer else 0.4
            CWmax = float(cw_coef * lai_med)  # mm
        else:
            CWmax = np.nan

        # --- seed SWinit prior (midway between PWP and FC) if we have both ---
        if np.isfinite(soildepth) and np.isfinite(ThetaFC) and np.isfinite(ThetaPWP):
            SWinit_prior = float(soildepth * (ThetaPWP + 0.5*(ThetaFC - ThetaPWP)))
        else:
            SWinit_prior = np.nan

        # --- site description (optional) ---
        sdesc = self.get_table('SITEDESCRIPTION', self.con)
        sd = sdesc.loc[sdesc['site_id'] == station_id]
        description = sd['description'].iat[0] if 'description' in sd.columns and len(sd) else None
        
        # --- SITES ---
        sites = self.get_table('SITES', self.con).copy()
        
        row = sites.loc[sites['site_id'] == station_id]
        row = row.iloc[0] if not row.empty else pd.Series(dtype=object)

        sites.columns = sites.columns.str.strip()

        row = sites.loc[sites['site_id'] == station_id]
        row = row.iloc[0] if not row.empty else pd.Series(dtype=object)

        def _get(row, key, default=None):
            if key in row.index and pd.notna(row[key]):
                return row[key]
            return default

        # Extract scalars (fall back to NaN/None)
        latitude   = float(_get(row, 'lat', np.nan))
        longitude  = float(_get(row, 'lon', np.nan))
        elevation  = float(_get(row, 'elevation_masl', np.nan))
        natVeg     = _get(row, 'natVegetation_code1', None)
        site_name  = _get(row, 'site', None)        # if present in your SITES table

        # Build a concise, human-readable summary
        parts = []
        if site_name: parts.append(str(site_name))
        if np.isfinite(elevation): parts.append(f"{int(round(elevation))} m a.s.l.")
        if np.isfinite(latitude) and np.isfinite(longitude):
            parts.append(f"{latitude:.4f}, {longitude:.4f}")
        if natVeg: parts.append(f"Nat. veg: {natVeg}")
        if description: parts.append(description)
        
        pretty_description = " | ".join(parts) if parts else None
        
        site_meta = {
            "site_id": int(station_id),
            "site_name": site_name,
            "latitude": latitude,
            "longitude": longitude,
            "elevation_masl": elevation,
            "natVegetation_code1": natVeg,
            "description_raw": description,
            "description": pretty_description,  # <- the composed one you can show/save
        }

        return {
            "site_id": int(station_id),
            "species": species,
            "is_conifer": bool(is_conifer),
            "lai_median": lai_med,
            "ThetaFC": ThetaFC,                 # m3/m3
            "ThetaPWP": ThetaPWP,               # m3/m3
            "soildepth": soildepth,             # mm
            "CWmax_est": CWmax,                 # mm
            "SWinit_prior": SWinit_prior,       # mm of water
            "description": site_meta,
        }
    
    def encode_doy(self, output):
        '''
        Encode cyclic features as day of year (DOY) for actual dates present in output
        :param output: DataFrame containing timestamps
        :return: Encoded sine and cosine features
        '''
        doy_sin = []
        doy_cos = []

        for date in output.index:
            day_of_year = date.dayofyear
            year = date.year
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):  # Leap year
                normalized_doy = day_of_year * (2. * np.pi / 366)
            else:
                normalized_doy = day_of_year * (2. * np.pi / 365)

            doy_sin.append(np.sin(normalized_doy))
            doy_cos.append(np.cos(normalized_doy))

        return np.array(doy_sin), np.array(doy_cos)
    
    @staticmethod
    def _to_py(v):
        # make JSON-safe (numpy -> python scalars)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.bool_,)):    return bool(v)
        return v

    def save_site_meta(self, meta: dict, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # ensure site_id present for filename
        sid = int(meta.get("site_id", meta.get("id", -1)))
        meta_clean = {k: self._to_py(v) for k, v in meta.items()}
        path = os.path.join(out_dir, f"site_{sid}_meta.json")
        with open(path, "w") as f:
            json.dump(meta_clean, f, indent=2)

    def get_location(self, station_id, doy=True): #rename function?
        """Bringing it all together 

        Args:
            station_id (_type_): _description_
            doy (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        
        #multisite as further argument?
        self.sid = station_id

        fapar = self.get_fapar(self.sid)
        clim = self.get_clim(self.sid)
        vpd = self.get_vpd(self.sid)
        gpp = self.get_gpp(self.sid)
        LE = self.get_et(self.sid)
        sw = self.get_swc(self.sid)

        # merge data
        op = self.shorten_merge(gpp, LE, clim, vpd, fapar, sw) 
        # Add site ID
        op['site_id'] = station_id

        # 7) Availability report / adjusted periods to full years
        output, report, adjusted_periods, _ = ProfoundData.availability_report_and_filter(
            op=op,
            station_id=station_id,
            essential_vars=['GPP','Precip','Tair','PAR','VPD','fapar','LE'],
            require_complete_years=True,
            verbose=True
        )
        if output is None:
            return None

        # latent heat to ET
        # latent heat vaporization ref: Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
        #                                                Kluwer Academic Publishers, Dordrecht, Netherlands

        # The calculation is based on physical laws of thermodynamics
        # 8) Convert LE → ET (mm/day)
        if 'LE' in output.columns and not output['LE'].isnull().all():
            # Latent heat to ET
            tair = output['Tair']
            LE = output['LE']                       # Latent heat flux in W/m²
            k1 = 2.501                              # MJ/kg at 0°C
            k2 = 0.00237                            # MJ/kg/°C
            lb = (k1 - k2 * tair) * 1e06            # Convert MJ/kg to J/kg
            output['ET'] = (LE / lb) * 86400        # Convert from W/m² to mm/day
            output = output.drop(['LE'], axis=1)
        else:
            print("Latent heat flux (LE) data is missing. Cannot calculate ET.")
            output['ET'] = np.nan  # Assign NaN or handle as appropriate

        
        # 9) Add DOY, CO2, time_idx, seasonality features
        output['DOY'] = output.index.dayofyear
        output['CO2'] = np.array([380]*len(output.index))

        # add time_steps difference required for pytorch forcasting dataset (time_idx)
        # Add time index for forecasting models
        time_steps =  output.index.to_series().diff()
        #remove first time step
        time_steps = time_steps.iloc[1:]
        time_steps = time_steps.cumsum()
        time_steps = time_steps.dt.days.astype(int)
        #add first time step with value 0
        time_steps = time_steps.reindex(pd.Index([str(output.index[0])]).append(time_steps.index),fill_value=0)
        time_steps = time_steps.reset_index(name='time_idx')
        #merge time _series
        output['time_idx'] = np.array((time_steps['time_idx']))
        
        if doy:
            #expects whole years !!!!!!!!
            sin_, cos_ = self.encode_doy(output)
            output['sin_idx'] = sin_
            output['cos_idx'] = cos_
        
        return output


    
if __name__=='__main__':
    save = os.getcwd()

    #change to current directory
    data_paths = os.path.join(save,'Data/ProfoundData.sqlite')

    path_INPUT = os.path.join(save,'Data/ProfoundData.zip') #put zip or extracted
    path_OUTPUT = os.path.join(save,"Data/preprocessed_data")
    print(path_OUTPUT)

    Data =ProfoundData(db_source=data_paths)

    #get 
    #possible station_id= 3,5,12,14,21 (id of site)
    """
    3   Bily Kritz #missing LE; therefore cannot caluculate ET. Can be used for just GPP predictions.
    5	collelongo		Italy	
    12	hyytiala		Finland	
    14	le_bray	    	France
    21	soro	    	Denmark
    
    """

    soro= Data.get_location(station_id=21)
    if soro is not None:
        output_file_path = os.path.join(path_OUTPUT, 'station_21.csv')
        soro.to_csv(output_file_path)
    else:
        print("No data available for Soro after processing.")

    hyytiala= Data.get_location(station_id=12)
    if hyytiala is not None:
        output_file_path = os.path.join(path_OUTPUT, 'station_12.csv')
        hyytiala.to_csv(output_file_path)
    else:
        print("No data available for hyytiala after processing.")

    collelongo= Data.get_location(station_id=5) 
    output_file_path = os.path.join(path_OUTPUT, 'station_5.csv')
    collelongo.to_csv(output_file_path)

    le_bray= Data.get_location(station_id=14)
    output_file_path = os.path.join(path_OUTPUT, 'station_14.csv')
    le_bray.to_csv(output_file_path)

    print('end of code')

