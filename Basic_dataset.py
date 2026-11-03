
import os
import re
import torch
import pandas as pd
#import preles
from torch.utils.data import Dataset, DataLoader
from utilis_shorts import json_open
from Data_preperation import ProfoundData
import warnings
import numpy as np
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union 


class Basic_dataset(Dataset):
    def __init__(self):
        super().__init__()
        # Stores normalisation of data in order to denormalise post experiment
        self.normalization: Optional[Dict[str, Any]] = None

    @staticmethod
    def _normalize_station_id(station_id):
        """Keep numeric IDs numeric-like, but allow string IDs such as 'CH-Dav'."""
        if isinstance(station_id, str):
            sid = station_id.strip()
            if sid.isdigit():
                return str(int(sid))
            return sid
        if isinstance(station_id, (int, np.integer)):
            return str(int(station_id))
        return str(station_id)

    @staticmethod
    def _normalize_source_tag(source_tag: Optional[str]) -> Optional[str]:
        if source_tag is None:
            return None
        tag = str(source_tag).strip().lower()
        return tag if tag else None

    @classmethod
    def _resolve_station_csv_path(
        cls,
        path_dir: str,
        station_id: str,
        source_tag: Optional[str] = None,
        source_preference: Sequence[str] = ("flx", "icos"),
    ) -> str:
        """
        Resolve station CSV path with source-aware logic.

        Resolution order:
        1) explicit source_tag -> station_<SITE>__<source>.csv (strict)
        2) legacy station_<SITE>.csv
        3) preferred source-tagged candidates by source_preference
        4) single tagged candidate if unique
        """
        sid = cls._normalize_station_id(station_id)
        src = cls._normalize_source_tag(source_tag)

        try:
            files = os.listdir(path_dir)
        except FileNotFoundError:
            files = []

        csv_files = [f for f in files if f.lower().endswith(".csv")]
        lower_map = {f.lower(): f for f in csv_files}

        station_re = re.compile(rf"^station_{re.escape(sid)}(?:__(?P<src>[^.]+))?\.csv$", re.IGNORECASE)
        station_candidates = []
        tagged_by_source: Dict[str, List[str]] = {}
        legacy_match = None
        for fname in csv_files:
            m = station_re.match(fname)
            if not m:
                continue
            station_candidates.append(fname)
            src_part = m.group("src")
            if src_part is None:
                legacy_match = fname
                continue
            src_norm = cls._normalize_source_tag(src_part)
            if src_norm is None:
                continue
            tagged_by_source.setdefault(src_norm, []).append(fname)

        if src is not None:
            wanted = tagged_by_source.get(src, [])
            if wanted:
                return os.path.join(path_dir, sorted(wanted)[0])
            expected = os.path.join(path_dir, f"station_{sid}__{src}.csv")
            candidates_txt = ", ".join(sorted(station_candidates)) if station_candidates else "none"
            raise FileNotFoundError(
                f"Requested source_tag='{src}' for station '{sid}' but file was not found at {expected}. "
                f"Available station candidates: {candidates_txt}"
            )

        if legacy_match is not None:
            return os.path.join(path_dir, legacy_match)

        pref_list = [cls._normalize_source_tag(s) for s in source_preference if cls._normalize_source_tag(s)]
        for pref in pref_list:
            matches = tagged_by_source.get(pref, [])
            if matches:
                return os.path.join(path_dir, sorted(matches)[0])

        all_tagged = sorted({fname for names in tagged_by_source.values() for fname in names})
        if len(all_tagged) == 1:
            return os.path.join(path_dir, all_tagged[0])

        if len(all_tagged) > 1:
            raise ValueError(
                f"Multiple source-tagged station files found for '{sid}': {all_tagged}. "
                "Set source_tag explicitly (e.g. 'flx' or 'icos')."
            )

        target_name = f"station_{sid}.csv"
        match = lower_map.get(target_name.lower())
        if match:
            return os.path.join(path_dir, match)

        target_path = os.path.join(path_dir, target_name)
        raise FileNotFoundError(
            f"Preprocessed data file not found at {target_path}. "
            f"Available station candidates for '{sid}': {sorted(station_candidates) if station_candidates else 'none'}"
        )

    def get_all_data(self,path_dir,station_id :Dict,
                     x_parameters: List, y_parameters:List, 
                     y_phys_para:Union[List, None]=None,
                     dataset= True, normalise=None, 
                     error=False, sparse_csv=False, 
                     previously_calibrated=True, 
                     error_norm = False,
                     sparse=None,
                     defer_sparse: bool = False,
                     source_tag: Optional[Union[str, Dict[str, str]]] = None,
                     source_preference: Sequence[str] = ("flx", "icos"),
                     normalization: Optional[Mapping[str, Any]] = None):

        """"
        get data from datsets while parameters are checked
        x_parameters: names/keys of input parameters
        y_paramters: names/keys of output parameters
        ....todo
        sparse: number when a number is picked, e.g. 7 means every seventh entry in dataset
        defer_sparse: keep every row and mark the sparse selection in a boolean
            `_sparse_keep` column instead of dropping rows, so a stateful process
            model can be simulated on the continuous daily series and thinned after.
        normalization: pre-fitted z-score constants to apply instead of self-fitting.
            `None` self-fits on this split's own dataframe, which for a test/val split
            leaks that split's statistics (including the observed target mean/sd) into
            the prediction path. Callers must hand down the train-fitted stats.
        """
        # Check config format for station_id; basically do you only have years given, or also start and stop day/month
        config_format = isinstance(next(iter(station_id.values())), dict) 

        #cfg['data::train::station_id']= {'21': {'startdate':[[2008,1,1]], 'stopdate': [[2008,12,31]]}}
        stations_keys = list(station_id.keys())
      
        for k, id in enumerate(stations_keys):
            station_source_tag = None
            if isinstance(source_tag, dict):
                station_source_tag = source_tag.get(id)
                if station_source_tag is None:
                    station_source_tag = source_tag.get(self._normalize_station_id(id))
            elif source_tag is not None:
                station_source_tag = source_tag
            
            if config_format:
                # ** Config-based format (has startdate & stopdate)**
                startindex = station_id[id]['startdate']
                stopindex = station_id[id]['stopdate']
                if station_source_tag is None and isinstance(station_id[id], dict):
                    station_source_tag = station_id[id].get("source_tag", station_id[id].get("source"))
            else:
                # ** Years-based format → Convert to startdate & stopdate**
                sorted_years = sorted(station_id[id])
                startindex = [[year, 1, 1] for year in sorted_years]  # Start of each year
                stopindex = [[year, 12, 31] for year in sorted_years]  # End of each year

            if k==0:
                self.data =  self.get_data(path_dir, 
                                           self._normalize_station_id(id), 
                                           dataset, # sparse_csv=sparse_csv, #redo sparse years.
                                           previously_calibrated=previously_calibrated,
                                           source_tag=station_source_tag,
                                           source_preference=source_preference, )
                  

                for l,index in enumerate(zip(startindex, stopindex)):
                    _, self.start = self.date_convert(index[0])
                    _, self.stop =self.date_convert(index[1])
                    if l==0: 
                        data_h = self.data.iloc[self.start:self.stop + 1]
                    else:
                        h_ = self.data.iloc[self.start:self.stop + 1]
                        data_h =  pd.concat([data_h, h_], ignore_index=True)
                self.data= data_h

                #check parameters
                if  y_phys_para!=None:
                    self.check_parameters(x_parameters, y_parameters, y_phys_para)
                    self.x_parameter = x_parameters
                    self.y_parameter = y_parameters
                    self.y_phy_para =  y_phys_para
                    self.data= self.data[x_parameters+y_parameters+y_phys_para]
                else:
                    self.check_parameters(x_parameters, y_parameters)
                    self.x_parameter = x_parameters
                    self.y_parameter = y_parameters
                    self.data= self.data[x_parameters+y_parameters]
                # keep site identity so temporal windows can be built per site; the
                # column subsetting above would otherwise discard it
                self.data = self.data.copy()
                self.data["_site_id"] = self._normalize_station_id(id)
                # Select sparse data if needed; thin each site's block once, before
                # concatenation, so earlier sites are not re-thinned per added site
                if sparse != None: #a number is expected
                    self.data = self._apply_sparse(self.data, sparse, defer_sparse)
            else:
                data_whole  =  self.get_data(path_dir,self._normalize_station_id(id),dataset, #sparse_csv=sparse_csv,
                                        previously_calibrated=previously_calibrated,
                                        source_tag=station_source_tag,
                                        source_preference=source_preference,
                                        )
                # startindex= station_id[id]['startdate']  
                # stopindex = station_id[id]['stopdate'] 

                for l,index in enumerate(zip(startindex, stopindex)):
                    _, self.start = self.date_convert(index[0], data=data_whole)
                    _, self.stop =self.date_convert(index[1], data=data_whole)
                    if l==0: 
                        data_h = data_whole.iloc[self.start:self.stop + 1]
                    else:
                        h_ = data_whole.iloc[self.start:self.stop + 1]
                        data_h =  pd.concat([data_h, h_], ignore_index=True)

                #check parameters
                if  y_phys_para!=None:
                    self.check_parameters(x_parameters, y_parameters, y_phys_para)
                    self.x_parameter = x_parameters
                    self.y_parameter = y_parameters
                    self.y_phy_para =  y_phys_para
                    data_h= data_h[x_parameters+y_parameters+y_phys_para]
                else: 
                    self.check_parameters(x_parameters, y_parameters)
                    self.x_parameter = x_parameters
                    self.y_parameter = y_parameters
                    data_h= data_h[x_parameters+y_parameters]
                data_h = data_h.copy()
                data_h["_site_id"] = self._normalize_station_id(id)
                if sparse != None: #a number is expected
                    data_h = self._apply_sparse(data_h, sparse, defer_sparse)
                self.data = pd.concat([self.data, data_h], ignore_index=True) #to do update

        if normalise=='z-score':
            self.normalise(normalization=normalization)
            if error:
                self.data,err_para = self.error_labels(self.data, real_label= y_parameters, preles_label= y_phys_para)
                self.check_parameters(err_para)
                self.err_para = err_para
        return  

    @staticmethod
    def _apply_sparse(block: pd.DataFrame, sparse: int, defer: bool) -> pd.DataFrame:
        """Keep every `sparse`-th row of one site's block, or (defer) mark those rows."""
        if not defer:
            return block.iloc[::sparse]
        block = block.copy()
        keep = np.zeros(len(block), dtype=bool)
        keep[::sparse] = True
        block["_sparse_keep"] = keep
        return block

    def get_data(self, path_dir, station_id, dataset=True,
                # sparse_csv=False,
                previously_calibrated=True,
                source_tag: Optional[str] = None,
                source_preference: Sequence[str] = ("flx", "icos"),
                # error_data=False, 
            ):
        
        """Get data from preprocessed csv file or profound database.

        Parameters:
            path_dir (str): Directory where the data is stored.
            station_id (int or list): Station ID(s) to retrieve data for.
            dataset (bool): If True, load preprocessed data from CSV.
            
        Returns:
            pd.DataFrame: DataFrame containing the requested data.
            If dataset is False, returns data from Profound database.
            
        """

        self.previously_calibrated = previously_calibrated # We can probably kick this out too now?
        station_id = self._normalize_station_id(station_id)

        if dataset and not previously_calibrated:
            csv_path = self._resolve_station_csv_path(
                path_dir=path_dir,
                station_id=station_id,
                source_tag=source_tag,
                source_preference=source_preference,
            )

            # Load preprocessed data
            data = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')

        
        elif not dataset:
            print(f"Opening Profound Database")
            zip_path = os.path.join(path_dir, "ProfoundData.zip")
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"ProfoundData.zip not found at {zip_path}")
            self.Profound = ProfoundData(zip_path) 
            if isinstance(station_id,list):
                station_id=station_id[0]
            station_id = self._normalize_station_id(station_id)
            if not str(station_id).isdigit():
                raise ValueError(
                    f"Profound DB lookup requires numeric station_id, got '{station_id}'."
                )
            data = self.Profound.get_location(int(station_id))

        return data
    
    def test_device(self, device):
        if device == True:
            if  torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.cuda = True
            else:
                self.cuda = False
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        return
    
    # candidate names for the per-site grouping key, in priority order
    _SITE_COL_CANDIDATES = ("_site_id", "site_id", "site", "station_id", "station")

    def _resolve_site_col(self):
        """Return the name of the site column in self.data, or None if absent."""
        for c in self._SITE_COL_CANDIDATES:
            if c in self.data.columns:
                return c
        return None

    def _make_shifter(self):
        """
        Build a shift function that never lets a window straddle a site boundary.

        A plain `df.shift(k)` over the concatenated multi-site frame splices the
        previous site's tail into the first k rows of the next site.  Those rows
        contain no NaN, so the dropna below does not remove them and the
        contamination is silent.  It scales with window length, so it is nearly
        invisible at in_wind=10 and severe at the ~180 d windows needed for
        long-memory (water accumulation) experiments.
        """
        site_col = self._resolve_site_col()

        if site_col is None:
            in_wind = int(getattr(self, "in_wind", 1) or 1)
            out_wind = int(getattr(self, "out_wind", 1) or 1)
            if max(in_wind, out_wind) > 1:
                warnings.warn(
                    "create_temporal_dataset: no site column found among "
                    f"{self._SITE_COL_CANDIDATES}; temporal windows are being built with a "
                    "global shift. If this frame holds more than one site, windows will "
                    "straddle site boundaries and silently mix sites.",
                    stacklevel=3,
                )
            return lambda frame, periods: frame.shift(periods)

        groups = self.data[site_col]
        return lambda frame, periods: frame.groupby(groups, sort=False).shift(periods)

    def create_temporal_dataset(self, physics=False, error=False, y_past_include = True):
        """ creates temporal dataset, depending on windows output and input size"""

        self.X_t_parameter = list()
        self.Y_t_parameter = list()
        _shift = self._make_shifter()

        if physics==True:
            self.Y_t_physics = list()
            names_in, cols_in= list(), pd.DataFrame()

            if error: #for adapted regularized
                self.err_t_y = list()
                self.err_t_x = list() 
                key_error= self.y_phy_para  #+self.err_para
                key = self.x_parameter #+ self.y_parameter
                data= self.data[key+key_error]
            elif y_past_include == False:
                key= self.x_parameter
                data = self.data[key]
            else:
                key= self.x_parameter+ self.y_parameter #only use real data, see if we can do it in every model
                data = self.data[key]

            for t_p in range (self.in_wind, 0,-1):
                #changes names of dataset depending on time
                names_in=[i+'(t-' + str(int(t_p)) +')' for i in key]
                if error:
                    names_err = [i+'(t-' + str(int(t_p)) +')' for i in key_error]
                    self.err_t_x.extend(names_err)
                    self.X_t_parameter.extend(names_in)
                    cols_in[names_in+names_err] = _shift(data, t_p)
                else:
                    self.X_t_parameter.extend(names_in)
                    #add data from the "past" 
                    cols_in[names_in]= _shift(data, t_p)
        else:
            names_in, cols_in= list(), pd.DataFrame()
            if y_past_include == False:
                key= self.x_parameter
                data = self.data[key]
            else:
                # the grouping key is bookkeeping, never a feature
                key = [k for k in self.data.keys() if k not in self._SITE_COL_CANDIDATES]
                data = self.data[key]
            for t_p in range (self.in_wind, 0,-1):
                #changes names of dataset depending on time
                names_in=[i+'(t-' + str(int(t_p)) +')' for i in key]
                self.X_t_parameter.extend(names_in)
                #add data from the "past" 
                cols_in[names_in]= _shift(data, t_p)
        
        #output window Y parameter
        if physics==True:
            names_out, names_out_p, cols_out= list(), list(), pd.DataFrame()

            if error:
                out_data = self.data[self.y_parameter+self.y_phy_para+self.err_para]
                self.err_t_y = list() 
            else:
                out_data = self.data[self.y_parameter+self.y_phy_para]

            for t_f in range(0,self.out_wind):
                names_out=[i+'(t+' + str(int(t_f)) +')' for i in self.y_parameter]
                names_out_p = [i+'(t+' + str(int(t_f)) +')' for i in self.y_phy_para]
                if error:
                    names_out_err = [i+'(t+' + str(int(t_f)) +')' for i in self.err_para]
                    cols_out[names_out+names_out_p+names_out_err ]=_shift(out_data, -t_f)
                    self.Y_t_physics.extend(names_out_p)
                    self.Y_t_parameter.extend(names_out)
                    self.err_t_y.extend(names_out_err)
                else: 
                    cols_out[names_out+names_out_p]=_shift(out_data, -t_f)
                    self.Y_t_physics.extend(names_out_p)
                    self.Y_t_parameter.extend(names_out)
        else:
            names_out, cols_out= list(), pd.DataFrame()
            out_data = self.data[self.y_parameter]
            for t_f in range(0,self.out_wind):
                names_out=[i+'(t+' + str(int(t_f)) +')' for i in list(out_data.keys())]
                cols_out[names_out]=_shift(out_data, -t_f)
                self.Y_t_parameter.extend(names_out)

        self.data =pd.concat([cols_in, cols_out], axis=1)
        self.data.dropna(inplace=True)
        self.data.index=[int(i) for i in range(0,len(self.data))]

        return 
     
    def rewrite_LSTM(self, physics=False, error=False):
        '''rewrite temporal data suitable for LSTM, in shape [sample, timesteps, features]'''
        #find temporal parameter x
        key_x = self.x_parameter #+ self.y_parameter # past y_parameter are not included
        self.x_LSTM =[[X_t for X_t in self.X_t_parameter if x in X_t] for x in key_x]

        #sort data in order and split in x and y
        new_data_x= [self.data[x].T for x in self.x_LSTM]
        new_data_x = np.array(new_data_x).transpose(2,1,0)

        if physics==True and error ==True:

            key_error_x = self.y_phy_para
            self.x_error_LSTM =[[X_t for X_t in self.err_t_x if x in X_t] for x in key_error_x]
            #sort data in order and split in x and y
            new_data_x_err = [self.data[x].T for x in self.x_error_LSTM]
            new_data_x_err = np.array(new_data_x_err).transpose(2,1,0)

        # y parameter 
        if physics==True and error ==False: #adds and sorts prameters of preles model
            self.y_LSTM = [[Y_t for Y_t in self.Y_t_parameter if x in Y_t] for x in self.y_parameter ]
            self.y_p_LSTM = [[Y_t for Y_t in self.Y_t_physics if x in Y_t] for x in self.y_phy_para]
            new_data_y= [self.data[y].T for y in self.y_LSTM]
            new_data_y = np.array(new_data_y).transpose(2,1,0) 
            new_data_y_p= [self.data[y].T for y in self.y_p_LSTM]
            new_data_y_p = np.array(new_data_y_p).transpose(2,1,0) 
            return new_data_x, new_data_y.squeeze(), new_data_y_p.squeeze()
        
        elif physics==True and error == True: #adds and sorts parameters of preles model. if error of preles model is included
            self.y_LSTM = [[Y_t for Y_t in self.Y_t_parameter if x in Y_t] for x in self.y_parameter ]
            self.y_p_LSTM = [[Y_t for Y_t in self.Y_t_physics if x in Y_t] for x in self.y_phy_para]
            self.y_err_LSTM = [[Y_t for Y_t in self.err_t_y if x in Y_t] for x in self.err_para]

            #sort data in LSTM order 
            new_data_y= [self.data[y].T for y in self.y_LSTM]
            new_data_y = np.array(new_data_y).transpose(2,1,0) 

            new_data_y_p= [self.data[y].T for y in self.y_p_LSTM]
            new_data_y_p = np.array(new_data_y_p).transpose(2,1,0) 

            new_data_y_err= [self.data[y].T for y in  self.y_err_LSTM]
            new_data_y_err = np.array(new_data_y_err).transpose(2,1,0) 

            return new_data_x, new_data_x_err, new_data_y.squeeze(), new_data_y_p.squeeze(),  new_data_y_err.squeeze()
        else:
            self.y_LSTM = [[Y_t for Y_t in self.Y_t_parameter if x in Y_t] for x in self.y_parameter]
            new_data_y= [self.data[y].T for y in self.y_LSTM]
            new_data_y = np.array(new_data_y).transpose(2,1,0) 
            return new_data_x, new_data_y.squeeze()
    
    def check_parameters(self, *parameters, error=False):
            # Check whether requested parameters exist in the loaded dataset.
            available = list(self.data.keys())
            available_set = set(available)

            for para in parameters:
                requested = list(para)
                missing = [p for p in requested if p not in available_set]
                if missing:
                    raise KeyError(
                        "Missing parameter columns in dataset. "
                        f"Missing: {missing}. \n"
                        f"Requested: {requested}. \n"
                        f"Available: {available}"
                    )

                if error:
                    # checks if order of real output data and process data are the same
                    for key_real, key_phy in zip(self.y_parameter, self.y_phy_para):
                        if key_real not in key_phy:
                            raise Exception("Order mismatch of Process and measured output data")
            return
    
    def date_convert(self,conv, data=None):
            """
            Converts a given [YYYY, MM, DD] format to the first and last index in the dataset.

            Parameters:
                conv (list): A list in the format [YYYY, MM, DD].
                data (pd.DataFrame, optional): The dataset to use. Defaults to self.data.

            Returns:
                Tuple: (Converted date, first index, last index in dataset).
            """
            try:
                # Use self.data if no dataset is provided
                if data is None:
                    data = self.data
                
                # Case 1: If input is already a Python `date` object
                if isinstance(conv, date):
                    time = conv
                    matching_indices = np.where(data.index.date == time)[0]

                    # Check if matching_indices is empty before converting to int
                    if matching_indices.size == 0:
                        raise ValueError(f"Date {time} not found in dataset.")

                    # Convert first occurrence to integer
                    indices = int(matching_indices[0])

                    return time, indices

                # Case 2: If input is a string ("YYYY-MM-DD")
                elif isinstance(conv, str):
                    time = pd.to_datetime(conv).date()
                    matching_indices = np.where(data.index.date == time)[0]

                    # Check if matching_indices is empty before converting to int
                    if matching_indices.size == 0:
                        raise ValueError(f"Date {time} not found in dataset.")

                    # Convert first occurrence to integer
                    indices = int(matching_indices[0])

                    return time, indices
                
                # # Case 3: If input is a list ([YYYY, MM, DD] or [[YYYY, MM, DD]])
                elif isinstance(conv, list):
                    if isinstance(conv[0], list):  # Handles nested lists
                        y, m, d = conv[0]
                    elif len(conv) == 3:  # Handles normal lists
                        y, m, d = conv
                    else:
                        raise ValueError(f"Invalid list format: {conv}")

                    time = date(y, m, d)
                    matching_indices = np.where(data.index.date == time)[0]

                    # Check if matching_indices is empty
                    if matching_indices.size == 0:
                        raise ValueError(f"Date {time} not found in dataset.")

                    # Convert first occurrence to integer
                    indices = int(matching_indices[0])

                    # # Debugging: Print the corresponding date in the dataset
                    # print(f"Date {time} found at index {indices}: {data.index[indices]}")

                    return time, indices
                
                else:
                    raise ValueError(f"Unsupported format for date conversion: {conv}")
                
            except Exception as e:
                raise RuntimeError(f"Error in date_convert: {e}")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _series_to_numpy(series: pd.Series) -> np.ndarray:
        return series.to_numpy(dtype=np.float32)

    def _fit_zscore_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        x_mean = self._series_to_numpy(data[self.x_parameter].mean())
        x_std = self._series_to_numpy(data[self.x_parameter].std(ddof=0).replace(0, 1))
        y_mean = self._series_to_numpy(data[self.y_parameter].mean())
        y_std = self._series_to_numpy(data[self.y_parameter].std(ddof=0).replace(0, 1))
        return {
            "kind": "z-score",
            # The constants are positional arrays, so carry the column names that were
            # used to fit them.  A stats dict fitted on the train split is applied to
            # test/val, and those splits resolve their own x/y column lists — if the
            # lists ever diverge (a stale `data::test::x_parameter`, a stale
            # `Basfor::y_phy_para`) the mismatch must fail loudly, not silently
            # normalise column i with column j's mean.
            "x_parameters": list(self.x_parameter),
            "y_parameters": list(self.y_parameter),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }

    def _check_stats_columns(self, normalization: Mapping[str, Any]) -> None:
        """Verify a handed-down stats dict was fitted on the same columns, in order.

        Older stats dicts (and hand-built ones) carry no column names; those are
        accepted unchecked for backwards compatibility.
        """
        for key, current in (("x_parameters", self.x_parameter),
                             ("y_parameters", self.y_parameter)):
            expected = normalization.get(key)
            if expected is None:
                continue
            if list(expected) != list(current):
                raise ValueError(
                    f"Normalisation statistics were fitted on different {key[0]} columns "
                    f"than this dataset uses.\n  fitted on: {list(expected)}\n"
                    f"  this split: {list(current)}\n"
                    "The z-score constants are positional, so applying them across "
                    "differing column lists would normalise the wrong columns."
                )

    def _apply_normalization(self, data: pd.DataFrame, normalization: Mapping[str, Any]) -> pd.DataFrame:
        self._check_stats_columns(normalization)

        out = data.copy()
        x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)
        x_std = np.asarray(normalization["x_std"], dtype=np.float32)
        y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)
        y_std = np.asarray(normalization["y_std"], dtype=np.float32)

        out.loc[:, self.x_parameter] = (out[self.x_parameter].to_numpy(dtype=np.float32) - x_mean) / x_std
        out.loc[:, self.y_parameter] = (out[self.y_parameter].to_numpy(dtype=np.float32) - y_mean) / y_std

        if hasattr(self, 'y_phy_para') and self.y_phy_para:
            # Process-model predictions are z-scored with the *observed* target's
            # constants, not their own, so the process model's bias and amplitude
            # error relative to observations survives the transform.
            x_cols = set(self.x_parameter)
            n_y = len(y_mean)
            phy_cols, phy_idx, unmatched = [], [], []
            for i, col in enumerate(self.y_phy_para):
                if col in x_cols:
                    # Already normalised by the x pass above (a residual model feeds
                    # <y>_pred in as an input).  Normalising again with y stats would
                    # apply the transform twice.
                    continue
                if i >= n_y:
                    # An extra prediction column (model::io::y_pred_additional_cols)
                    # with no matching target — there is no meaningful constant for it.
                    unmatched.append(col)
                    continue
                phy_cols.append(col)
                phy_idx.append(i)

            if unmatched:
                warnings.warn(
                    f"No target statistics for process-prediction column(s) {unmatched}; "
                    "leaving them in physical units. They have no counterpart in "
                    f"y_parameter ({list(self.y_parameter)}), so z-scoring them with a "
                    "target's mean/sd would be meaningless.",
                    stacklevel=3,
                )

            if phy_cols:
                idx = np.asarray(phy_idx, dtype=int)
                out.loc[:, phy_cols] = (
                    out[phy_cols].to_numpy(dtype=np.float32) - y_mean[idx]
                ) / y_std[idx]
        return out

    def denormalize_targets(self, values, normalization: Optional[Mapping[str, Any]] = None):
        stats = normalization if normalization is not None else self.normalization
        if stats is None:
            return values
        y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
        y_std = np.asarray(stats["y_std"], dtype=np.float32)
        if isinstance(values, pd.DataFrame):
            out = values.copy()
            out.loc[:, self.y_parameter] = out[self.y_parameter].to_numpy(dtype=np.float32) * y_std + y_mean
            return out
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim >= 2 and arr.shape[1] != y_std.shape[0]:
            n = arr.shape[1]
            return arr * y_std[:n] + y_mean[:n]
        return arr * y_std + y_mean

    def normalise(self, data=None, error = False, error_norm = False, normalization: Optional[Mapping[str, Any]] = None):
        target = self.data if data is None else data
        stats = dict(normalization) if normalization is not None else self._fit_zscore_stats(target)
        normalized = self._apply_normalization(target, stats)

        if data is None:
            self.data = normalized
            self.normalization = stats
            return self.data

        return normalized
        
    def error_labels(self,data, real_label:list, preles_label:list):
        ''' calculate error of process model, with respect to normalized data'''
        new_para= []
        for k in zip(real_label, preles_label):
            name= k[0]+ '_error'
            new_para.append(name)

            data[name]= np.abs(data[k[0]]- data[k[1]])
        return data, new_para

if __name__=='__main__':
    from Config_Files.config_spatial import build_config     ## explicitly set which experiment configuration file you wish to use
    
    config = build_config()
    save = config["cfg::save"]

    # save = os.getcwd()
    # config_path = os.path.join(save,'config.json')
    # config = json_open(config_path)
