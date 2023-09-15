## Note, the code used for preparing tidal currents data is not included but is nearly identical (just repeating things twice)


class Response_Framework:
  def __init__(self, ts, lat, lon, mult_val,include_real = None, datatype='currents', path = None):
      self.ts = ts
      self.N = len(ts)
      self.height = self.ts.values
      self.lat = lat
      self.lon = lon
      self.mult_val = mult_val
      self.time_step = self.ts.index.freq.delta.seconds /86400 ##  Cycles per day
      self.total_days = (self.ts.index.max() - self.ts.index.min()).total_seconds() / 86400
      self.path = path
      self.include_real = include_real
      self.n = 2*self.total_days/self.time_step
      self.M_E = 5.9722*10**24 ## Mass Earth
      self.M_M = 7.3 *10**22 ## Mass Moon
      self.M_S = 1.989 * 10**30 ## Mass Sun
      self.E_r = 6371.01*10**3 ## Earth Radius (denoted a in Response Paper)
      self.solar_constant = 1.946/100  ## Note in CM (Trying to convert to meters)
      self.degrees = range(1,3) ## Number of spherical harmonic degrees to compute.
      self.datatype = datatype

  def Spherical_Harmonic(self,degree, order):
    '''PreDefined Legendre Polynomials for speed:

       Stored in a nested list defined by:
       P_vals where i is the degree of the associated Legendre polynomials.
       Each value in the list is the Legendre polynomial of degree i and order
       equal to its index value in the list.

       Theta = geographical colatitude of station
       Lambda = longitude of station

       For Example:
       P_2 = [P_2^0,P_2^1, P_2^2] '''
    P_0 = [1]
    P_1 = [cosd(self.lat), sind(self.lat)]
    P_2 = [(1.5)*cosd(self.lat)**2 - .5, 3*sind(self.lat)*cosd(self.lat), 3*sind(self.lat)**2]
    P_3 = [(2.5)*cosd(self.lat)**3 - 1.5*cosd(self.lat), 1.5*sind(self.lat) * (5*cosd(self.lat)**2 -1), 15*sind(self.lat)**2 * cosd(self.lat), 15* sind(self.lat)**3]
    P_vals = [P_0,P_1,P_2,P_3]
    try:
      Y = (-1)**(order) *(2*degree + 1 / (4*np.pi))**(1/2) * (math.factorial(degree-order)/math.factorial(degree+order)) *cosd(self.lat) * np.exp(1j*order*self.lon)*P_vals[degree][order]
      return Y
    except ValueError:
      print("Order must be less than or equal to the degree")


  def Radiational(self, degree, order,filtered = False):
    '''Computes the Global Radiation Function for all zenith angles'''
    degree= int(degree)
    if degree > 2:
      raise ValueError('Currently only equipped to handle k = 0,1,2.')
    parallax = 1.0/23455.0
    k_n = [1/4 + (1/6)*parallax, (1/2) + (3/8)*parallax, (5/16) + (1/3)*parallax]

    SPH = self.Spherical_Harmonic(degree, order)
    k = k_n[degree]
    rads = pd.DataFrame(self.astro['zenith_sun'])
    rads['rad'] = self.solar_constant * (self.mean_r_sun / self.astro['station2sun']) * k * self.astro['mu_sun']**degree
    rads.loc[(rads['zenith_sun'] >=90) & (rads['zenith_sun'] <= 180), 'rad'] = 0
    return rads['rad'].to_numpy() / SPH

  def Gravitational(self, degree, order, filtered=False):
    """Computes the Gravitational Input function for each t of the observed time series:

       Assumes that the combined gravitational input function is equal to the sum of the
       lunar and solar input functions.

       **Currently using predefined Legendre functions as defined in Munk & Carwright
       Inputs
       --------
       degree: int (0,3)

       Returns
       --------
       : DataFrame

       Sum of gravitational forces from the moon and sun at specified degree. """
    degree= int(degree)
    if degree > 3:
       raise ValueError('Currently only equipped to handle k = 0,1,2,3.')

    SPH = self.Spherical_Harmonic(degree, order)
    ## Computing associated k values
    K_n_Moon = self.E_r * (self.M_M / self.M_E) * (self.E_r/ self.astro['earth2moon'])**(degree+1)
    K_n_Sun = self.E_r * (self.M_S / self.M_E) * (self.E_r/ self.astro['earth2sun'])**(degree+1)

    grav_Moon= K_n_Moon * (self.mean_r_moon/ self.astro['earth2moon'])**(degree+1) * self.Lunar_Legendre[degree]
    grav_Sun= K_n_Sun * (self.mean_r_sun/ self.astro['earth2sun'])**(degree+1)  * self.Solar_Legendre[degree]

    if self.datatype == 'currents':
      grav_Moon_U  = grav_Moon *cosd(self.astro['azimuth_moon'])
      grav_Moon_V  = grav_Moon *sind(self.astro['azimuth_moon'])
      grav_Sun_U  = grav_Sun *cosd(self.astro['azimuth_sun'])
      grav_Sun_V  = grav_Sun *sind(self.astro['azimuth_sun'])
      return (grav_Moon_U + grav_Sun_U).to_numpy() / SPH, (grav_Moon_V + grav_Sun_V).to_numpy() / SPH

    filtered_series =  (grav_Moon + grav_Sun).to_numpy()

    return filtered_series / SPH

  def Global_Tide(self):
    ''' Computes global tide function according to specified input functions
        Note** Radiational function commences with P_1(mu) and gravitational
        commences at P_2(mu)

        ** Note currently only setup to handle radiational and gravitational
        input functions (dictionary initialization will need to be adjusted)

        Returns
        --------
        : Dictionary
        Format: {'Radiational':{degree: Associated Radiational Input Values,...}
                 'Gravitational':{degree: Associated Gravitational Input Values,...}}

        Values are numpy arrays containing the instantaneous values of the input function for each time.

        returned as: self.global_tide'''
    self.Astro()
    ## Creating Helper_df ## Originally was checking the datatype to see if it was 'sim'. Assuming this has been resolved.
    self.helper_df = self.padded_ts
    if self.datatype != 'currents':
      self.global_tide = {'Radiational':{i: {k: 0 for k in range(1,i+1)} for i in range(1,3)}, 'Gravitational':{i:{k: 0 for k in range(1,i+1)} for i in range(2,4)}}
      for i in tqdm(self.global_tide, leave=False, desc='Computing Global Tide Function'):
        for j in self.global_tide[i]:
          if i == 'Radiational':
            for k in range(j+1):
              self.global_tide[i][j][k] = self.Radiational(degree = j, order = k, filtered = filter) ## Computes the Radiational Function at each Solar Zenith Angle For degree j.
          elif i == 'Gravitational':
            for k in range(j+1):
              self.global_tide[i][j][k] = self.Gravitational(degree = j, order = k, filtered = filter) ## Computes the Gravitational Function at each Solar/Lunar Zenith Angle  for degree j
          else:
            print("RTide is currently only equipped to handle gravitational and radiational inputs")
    else:
      self.global_tide = {'Radiational':{i: {k: 0 for k in range(1,i+1)} for i in range(1,3)}, 'Gravitational':{i:{k: {'u':0, 'v':0} for k in range(1,i+1)} for i in range(2,4)}}
      for i in tqdm(self.global_tide, leave=False, desc='Computing Global Tide Function'):
        for j in self.global_tide[i]:
          if i == 'Radiational':
            for k in range(j+1):
              self.global_tide[i][j][k] = self.Radiational(degree = j, order = k, filtered = filter) ## Computes the Radiational Function at each Solar Zenith Angle For degree j.
          elif i == 'Gravitational':
            for k in range(1,j+1):
              self.global_tide[i][j][k]['u'], self.global_tide[i][j][k]['v'] = self.Gravitational(degree = j, order = k, filtered = filter) ## Computes the Gravitational Function at each Solar/Lunar Zenith Angle  for degree j
          else:
            print("RTide is currently only equipped to handle gravitational and radiational inputs")


  def Astro(self):
    """Computes the associated astronomical positions and angles of the Moon and Sun
     relative to the station and center of the Earth."""
    ## Converting time index to appropriate format
    tindex = self.padded_ts.index.to_pydatetime()
    tscale = load.timescale()
    times = tscale.utc(tindex)
    ## Loading planet objects from skyfield
    planets = load('de421.bsp')
    earth, moon, sun = planets['earth'], planets['moon'], planets['sun']

    ## Computing Distances:
    earth_moon = [(moon.at(t) - earth.at(t)).distance().m for t in tqdm(times, leave=False, desc='Computing Distances From Earth to Moon')]## Returns Moons distance in meters
    earth_sun = [(sun.at(t) - earth.at(t)).distance().m for t in tqdm(times, leave=False, desc='Computing Distances From Earth to Sun')] ## Returns Suns distance in meters

    ## Computing Angles:
    moon_stat_dist = []
    sun_stat_dist = []
    zenith_moon = []
    zenith_sun = []
    azimuth_moon = []
    azimuth_sun = []
    for t in tqdm(times, leave=False, desc='Computing Distances/Angles From Station to Moon/Sun'):
      station_location = earth + wgs84.latlon(self.lat, self.lon)
      astro_moon = station_location.at(t).observe(moon)
      astro_sun = station_location.at(t).observe(sun)

      app_moon = astro_moon.apparent()
      app_sun = astro_sun.apparent()


      alt_moon, az_moon, stat_dist_moon = app_moon.altaz()
      alt_sun, az_sun, stat_dist_sun = app_sun.altaz()


      ## Zenith angle is equal to +90 for alt.
      zenith_moon.append(90 - alt_moon.degrees)
      zenith_sun.append(90 - alt_sun.degrees)

      azimuth_moon.append(az_moon.degrees)
      azimuth_sun.append(az_sun.degrees)

      moon_stat_dist.append(stat_dist_moon.m)
      sun_stat_dist.append(stat_dist_sun.m)
    self.mu_moon = cosd(zenith_moon) ## Mu Moon
    self.mu_sun = cosd(zenith_sun) ## Mu Sun
    self.alpha = zenith_sun ## Solar Zenith Angle
    self.mean_r_sun = np.sum(earth_sun) / len(earth_sun) ## Average Distance to sun
    self.mean_r_moon = np.sum(earth_moon) / len(earth_moon)
    self.station2sun = sun_stat_dist ## Distance from Station to Sun

    ## Associated Legendre Functions
    self.Solar_Legendre =  [1,self.mu_sun, [(3/2)*mew**2 - (1/2) for mew in self.mu_sun], [(5/2)*mew**3 - (3/2)*mew for mew in self.mu_sun]]
    self.Lunar_Legendre = [1,self.mu_moon, [(3/2)*mew**2 - (1/2) for mew in self.mu_moon], [(5/2)*mew**3 - (3/2)*mew for mew in self.mu_moon]]
    self.astro = pd.DataFrame({'station2moon':moon_stat_dist, 'station2sun': sun_stat_dist, 'earth2moon': earth_moon, 'earth2sun': earth_sun, 'zenith_moon': zenith_moon, 'zenith_sun': zenith_sun, 'mu_sun': cosd(zenith_sun), 'azimuth_moon': azimuth_moon, 'azimuth_sun': azimuth_sun}, index = self.padded_ts.index)

  def Prep(self):
    ## First Run basic utide to figure out how many constits it can handle
    if self.datatype != 'currents':
      short_utide = utide.solve(
      self.ts[:7*24*self.mult_val].index,
      self.ts[:7*24*self.mult_val].observations,
      lat=self.lat,
      method="ols",
      conf_int="MC",
      verbose=False,
      trend = False
      )
    else:
      short_utide = utide.solve(
      self.ts[:7*24*self.mult_val].index,
      self.ts[:7*24*self.mult_val].u,
      self.ts[:7*24*self.mult_val].v,
      lat=self.lat,
      method="ols",
      conf_int="MC",
      verbose=False,
      trend = False
      )
    ## Sorting Lags and Determining Largest Constit
    ## I think it should already have the correct units.
    nonuni = []
    for i,j in enumerate(np.sort(short_utide['aux']['frq'])):
      if (len(nonuni) != 0) and (nonuni[-1] == -int(2*np.pi / (4*j))):
        pass
      else:
        nonuni.append(-int(2*np.pi / (4*j)))
    nonuni.append(0)
    for i in np.flip(np.sort(short_utide['aux']['frq'])):
      if nonuni[-1] == int(2*np.pi / (4*i)):
        pass
      else:
        nonuni.append(int(2*np.pi / (4*i)))

    self.nonunilags = nonuni
    if not self.include_real:
      self.max_lag = max(nonuni)*self.mult_val ## Just checking
    else:
      self.max_lag = max(self.include_real, max(nonuni))*self.mult_val
    ## Getting Associated Constituent Names:
    sorts = np.argsort(short_utide['aux']['frq'])
    constits = list(short_utide['name'])
    new_vals = [constits[i] for i in sorts]
    new_vals2 = [constits[i] for i in np.flip(sorts)]
    new_vals.append('real')
    self.const_names = new_vals+ new_vals2


    ## Pad out to that point
    self.padded_ts = pad_dataframe(self.ts, int(self.max_lag / self.mult_val))

  def Prepare_Dfs(self, reduce = False):
    """
    Prepares training data by combining lagged input functions into one df.
    Precomputed global_tide input functions are done first, then a second
    loop incorporates any real lagged values or custom forcing depending
    on the end user choice.

    Nan values do not need to be considered as they will just be dropped
    before training.

    **Definitely not the most efficient, but DF format was helpful for sanity
    checking the results

    returns: DataFrame

    Containing
    """
    ## Let's check if this exists already:
    if self.path:
      try:
        prepped_dfs = pd.read_csv(f'{self.path}global_tide.csv',index_col=0)
        prepped_dfs.index = self.ts.index
        self.prepped_dfs = prepped_dfs
      except:
        self.Prep()
        self.Global_Tide()
        grav_names = []
        grav_values = []
        rad_values = []
        rad_names = []
        for i in self.global_tide['Radiational']:
          for j in self.global_tide['Radiational'][i]:
            rad_values.append(self.global_tide['Radiational'][i][j].real)
            rad_names.append('Radiational_' + str(i) +'^' +str(j))
        for i in self.global_tide['Gravitational']:
          for j in self.global_tide['Gravitational'][i]:
            grav_names.append('Gravitational_' + str(i)+'^' + str(j))
            grav_values.append(self.global_tide['Gravitational'][i][j].real)

        ## Initializing list of dfs that will be concatenated
        dfs = [self.ts]
        ## If not using PCA to reduce number of inputs
        if not reduce:
          ## Computing lags for gravitational inputs
          for i,name in enumerate(grav_names):
              gravs_real = [pd.DataFrame({name +'_'+ str(lag)+'_'+self.const_names[z]: grav_values[i][self.max_lag +lag: lag -self.max_lag].real}, index = self.ts.index) for z,lag in enumerate(self.nonunilags)]
              for x in range(len(gravs_real)):
                dfs.append(gravs_real[x])
          ## Computing lags for radiational inputs
          for i,name in enumerate(rad_names):
              rads_real = [pd.DataFrame({name+'_'+ str(lag)+'_'+self.const_names[z]: rad_values[i][self.max_lag + lag :lag -self.max_lag].real}, index = self.ts.index) for z,lag in enumerate(self.nonunilags)]
              for x in range(len(rads_real)):
                dfs.append(rads_real[x])
        else:
          ## Using PCA to reduce inputs
          ## Computing PCA for both Gravitational and Radiational Inputs
          gravs_red = dim_reduc(grav_values)
          print('gravs_red', gravs_red)
          rads_red = dim_reduc(rad_values)
          ## Computing lags for reduced gravtiational
          for i in range(len(gravs_red)):
              gravs_red_real = [pd.DataFrame({f'gravitational_reduced_{i}'+ str(lag)+'_'+self.const_names[z]: gravs_red[j][self.max_lag +lag: lag -self.max_lag].real}, index = self.ts.index) for z,lag in enumerate(self.nonunilags)]
              ## Adding to big dfs
              for x in range(len(gravs_red_real)):
                dfs.append(gravs_red_real[x])
          ## Computing lags for reduced radiational
          for i in range(len(rads_red)):
              rads_red_real = [pd.DataFrame({f'radiational_reduced_{i}'+ str(lag)+'_'+self.const_names[z]: rads_red[i][self.max_lag +lag: lag -self.max_lag].real}, index = self.ts.index) for z,lag in enumerate(self.nonunilags)]
              ## Adding to big dfs
              for x in range(len(rads_red_real)):
                dfs.append(rads_red_real[x])

        ## Added functionality for multivariate and localized forecasting.
        for i,col in enumerate(self.ts.columns):
          if col == 'observations':
            ## Functionality to include actual observations in training set lagged by "include_real" hours.
            ## For example if I am interested in making 24 hour ahead forecasts I can include the observation from 24 hours
            ## before to improve localization
            if self.include_real != None:
              negs = -self.include_real*self.mult_val
              cust_real = pd.DataFrame({'real_lagged'+'_'+ str(negs): self.padded_ts[col][self.max_lag + negs :negs -self.max_lag].to_numpy()}, index = self.ts.index)
              dfs.append(cust_real)
            else:
              pass
          else:
            ## Used for including lagged inputs for custom input functions
            custom_input = [pd.DataFrame({col+'_'+ str(lag)+'_'+self.const_names[z]: self.padded_ts[col][self.max_lag + lag :lag -self.max_lag].to_numpy()}, index = self.ts.index) for z,lag in enumerate(self.nonunilags)]
            for x in range(len(custom_input)):
              dfs.append(custom_input[x])

        prepped_dfs = pd.concat(dfs, join='outer', axis=1)
        self.prepped_dfs = prepped_dfs
        if self.path:
          self.prepped_dfs.to_csv(f'{self.path}global_tide.csv')
