# Core - Nixtla

- 元URL: https://nixtlaverse.nixtla.io/neuralforecast/core.html

NeuralForecast contains two main components, PyTorch implementations deep learning predictive models, as well as parallelization and distributed computation utilities. The first component comprises low-level PyTorch model estimator classes like `models.NBEATS` and `models.RNN`. The second component is a high-level `core.NeuralForecast` wrapper class that operates with sets of time series data stored in pandas DataFrames.


[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L217)
### [​](#neuralforecast)NeuralForecast


> CopyAsk AI NeuralForecast (models:List[Any], freq:Union[str,int],
>                  local_scaler_type:Optional[str]=None)


*The `core.StatsForecast` class allows you to efficiently fit multiple
[`NeuralForecast`](https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast)
models for large sets of time series. It operates with pandas DataFrame
`df` that identifies series and datestamps with the `unique_id` and `ds`
columns. The `y` column denotes the target time series variable.*
**Type****Default****Details**modelsListInstantiated `neuralforecast.models`   
see [collection here](https://nixtla.github.io/neuralforecast/models.html).freqUnionFrequency of the data. Must be a valid pandas or polars offset alias, or an integer.local_scaler_typeOptionalNoneScaler to apply per-serie to all features before fitting, which is inverted after predicting.  
Can be ‘standard’, ‘robust’, ‘robust-iqr’, ‘minmax’ or ‘boxcox’**Returns****NeuralForecast****Returns instantiated [`NeuralForecast`](https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast) class.**

[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L418)
### [​](#neuralforecast-fit)NeuralForecast.fit


> CopyAsk AI NeuralForecast.fit (df:Union[pandas.core.frame.DataFrame,polars.dataframe
>                      .frame.DataFrame,neuralforecast.compat.SparkDataFrame
>                      ,Sequence[str],NoneType]=None, static_df:Union[pandas
>                      .core.frame.DataFrame,polars.dataframe.frame.DataFram
>                      e,neuralforecast.compat.SparkDataFrame,NoneType]=None
>                      , val_size:Optional[int]=0,
>                      use_init_models:bool=False, verbose:bool=False,
>                      id_col:str='unique_id', time_col:str='ds',
>                      target_col:str='y', distributed_config:Optional[neura
>                      lforecast.common._base_model.DistributedConfig]=None,
>                      prediction_intervals:Optional[neuralforecast.utils.Pr
>                      edictionIntervals]=None)


*Fit the core.NeuralForecast.
Fit `models` to a large set of time series from DataFrame `df`. and
store fitted models for later inspection.*
**Type****Default****Details**dfUnionNoneDataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.  
If None, a previously stored dataset is required.static_dfUnionNoneDataFrame with columns [`unique_id`] and static exogenous.val_sizeOptional0Size of validation set.use_init_modelsboolFalseUse initial model passed when NeuralForecast object was instantiated.verboseboolFalsePrint processing steps.id_colstrunique_idColumn that identifies each serie.time_colstrdsColumn that identifies each timestep, its values can be timestamps or integers.target_colstryColumn that contains the target.distributed_configOptionalNoneConfiguration to use for DDP training. Currently only spark is supported.prediction_intervalsOptionalNoneConfiguration to calibrate prediction intervals (Conformal Prediction).**Returns****NeuralForecast****Returns [`NeuralForecast`](https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast) class with fitted `models`.**

[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L796)
### [​](#neuralforecast-predict)NeuralForecast.predict


> CopyAsk AI NeuralForecast.predict (df:Union[pandas.core.frame.DataFrame,polars.dataf
>                          rame.frame.DataFrame,neuralforecast.compat.SparkD
>                          ataFrame,NoneType]=None, static_df:Union[pandas.c
>                          ore.frame.DataFrame,polars.dataframe.frame.DataFr
>                          ame,neuralforecast.compat.SparkDataFrame,NoneType
>                          ]=None, futr_df:Union[pandas.core.frame.DataFrame
>                          ,polars.dataframe.frame.DataFrame,neuralforecast.
>                          compat.SparkDataFrame,NoneType]=None,
>                          verbose:bool=False, engine=None,
>                          level:Optional[List[Union[int,float]]]=None,
>                          quantiles:Optional[List[float]]=None,
>                          **data_kwargs)


*Predict with core.NeuralForecast.
Use stored fitted `models` to predict large set of time series from
DataFrame `df`.*
**Type****Default****Details**dfUnionNoneDataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.  
If a DataFrame is passed, it is used to generate forecasts.static_dfUnionNoneDataFrame with columns [`unique_id`] and static exogenous.futr_dfUnionNoneDataFrame with [`unique_id`, `ds`] columns and `df`’s future exogenous.verboseboolFalsePrint processing steps.engineNoneTypeNoneDistributed engine for inference. Only used if df is a spark dataframe or if fit was called on a spark dataframe.levelOptionalNoneConfidence levels between 0 and 100.quantilesOptionalNoneAlternative to level, target quantiles to predict.data_kwargsVAR_KEYWORDExtra arguments to be passed to the dataset within each model.**Returns****pandas or polars DataFrame****DataFrame with insample `models` columns for point predictions and probabilistic  
predictions for all fitted `models`. **

[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L1089)
### [​](#neuralforecast-cross-validation)NeuralForecast.cross_validation


> CopyAsk AI NeuralForecast.cross_validation (df:Union[pandas.core.frame.DataFrame,pol
>                                   ars.dataframe.frame.DataFrame,NoneType]=
>                                   None, static_df:Union[pandas.core.frame.
>                                   DataFrame,polars.dataframe.frame.DataFra
>                                   me,NoneType]=None, n_windows:int=1,
>                                   step_size:int=1,
>                                   val_size:Optional[int]=0,
>                                   test_size:Optional[int]=None,
>                                   use_init_models:bool=False,
>                                   verbose:bool=False,
>                                   refit:Union[bool,int]=False,
>                                   id_col:str='unique_id',
>                                   time_col:str='ds', target_col:str='y', p
>                                   rediction_intervals:Optional[neuralforec
>                                   ast.utils.PredictionIntervals]=None, lev
>                                   el:Optional[List[Union[int,float]]]=None
>                                   , quantiles:Optional[List[float]]=None,
>                                   **data_kwargs)


*Temporal Cross-Validation with core.NeuralForecast.
`core.NeuralForecast`’s cross-validation efficiently fits a list of
NeuralForecast models through multiple windows, in either chained or
rolled manner.*
**Type****Default****Details**dfUnionNoneDataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.  
If None, a previously stored dataset is required.static_dfUnionNoneDataFrame with columns [`unique_id`] and static exogenous.n_windowsint1Number of windows used for cross validation.step_sizeint1Step size between each window.val_sizeOptional0Length of validation size. If passed, set `n_windows=None`.test_sizeOptionalNoneLength of test size. If passed, set `n_windows=None`.use_init_modelsboolFalseUse initial model passed when object was instantiated.verboseboolFalsePrint processing steps.refitUnionFalseRetrain model for each cross validation window.  
If False, the models are trained at the beginning and then used to predict each window.  
If positive int, the models are retrained every `refit` windows.id_colstrunique_idColumn that identifies each serie.time_colstrdsColumn that identifies each timestep, its values can be timestamps or integers.target_colstryColumn that contains the target.prediction_intervalsOptionalNoneConfiguration to calibrate prediction intervals (Conformal Prediction).levelOptionalNoneConfidence levels between 0 and 100.quantilesOptionalNoneAlternative to level, target quantiles to predict.data_kwargsVAR_KEYWORDExtra arguments to be passed to the dataset within each model.**Returns****Union****DataFrame with insample `models` columns for point predictions and probabilistic  
predictions for all fitted `models`. **

[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L1262)
### [​](#neuralforecast-predict-insample)NeuralForecast.predict_insample


> CopyAsk AI NeuralForecast.predict_insample (step_size:int=1,
>                                   level:Optional[List[Union[int,float]]]=N
>                                   one,
>                                   quantiles:Optional[List[float]]=None)


*Predict insample with core.NeuralForecast.
`core.NeuralForecast`’s `predict_insample` uses stored fitted `models`
to predict historic values of a time series from the stored dataframe.*
**Type****Default****Details**step_sizeint1Step size between each window.levelOptionalNoneConfidence levels between 0 and 100.quantilesOptionalNoneAlternative to level, target quantiles to predict.**Returns****pandas.DataFrame****DataFrame with insample predictions for all fitted `models`. **

[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L1433)
### [​](#neuralforecast-save)NeuralForecast.save


> CopyAsk AI NeuralForecast.save (path:str, model_index:Optional[List]=None,
>                       save_dataset:bool=True, overwrite:bool=False)


*Save NeuralForecast core class.
`core.NeuralForecast`’s method to save current status of models,
dataset, and configuration. Note that by default the `models` are not
saving training checkpoints to save disk memory, to get them change the
individual model `**trainer_kwargs` to include
`enable_checkpointing=True`.*
**Type****Default****Details**pathstrDirectory to save current status.model_indexOptionalNoneList to specify which models from list of self.models to save.save_datasetboolTrueWhether to save dataset or not.overwriteboolFalseWhether to overwrite files or not.
CopyAsk AI```
/opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/fastcore/docscrape.py:230: UserWarning: potentially wrong underline length... 
Parameters 
----------- in 
Load NeuralForecast
...
  else: warn(msg)
```



[source](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py#L1549)
### [​](#neuralforecast-load)NeuralForecast.load


> CopyAsk AI NeuralForecast.load (path, verbose=False, **kwargs)


*Load NeuralForecast
`core.NeuralForecast`’s method to load checkpoint from path.*
**Type****Default****Details**pathstrDirectory with stored artifacts.verboseboolFalsekwargsVAR_KEYWORDAdditional keyword arguments to be passed to the function  
`load_from_checkpoint`.**Returns****NeuralForecast****Instantiated [`NeuralForecast`](https://nixtlaverse.nixtla.io/neuralforecast/core.html#neuralforecast) class.**
CopyAsk AI```
# Test predict_insample step_size

h = 12
train_end = AirPassengersPanel_train['ds'].max()
sizes = AirPassengersPanel_train['unique_id'].value_counts().to_numpy()
for step_size, test_size in [(7, 0), (9, 0), (7, 5), (9, 5)]:
    models = [NHITS(h=h, input_size=12, max_steps=1)]
    nf = NeuralForecast(models=models, freq='M')
    nf.fit(AirPassengersPanel_train)
    # Note: only apply set_test_size() upon nf.fit(), otherwise it would have set the test_size = 0
    nf.models[0].set_test_size(test_size)
    
    forecasts = nf.predict_insample(step_size=step_size)
    last_cutoff = train_end - test_size * pd.offsets.MonthEnd() - h * pd.offsets.MonthEnd()
    n_expected_cutoffs = (sizes[0] - test_size - nf.h + step_size) // step_size

    # compare cutoff values
    expected_cutoffs = np.flip(np.array([last_cutoff - step_size * i * pd.offsets.MonthEnd() for i in range(n_expected_cutoffs)]))
    actual_cutoffs = np.array([pd.Timestamp(x) for x in forecasts[forecasts['unique_id']==nf.uids[1]]['cutoff'].unique()])
    np.testing.assert_array_equal(expected_cutoffs, actual_cutoffs, err_msg=f"{step_size=},{expected_cutoffs=},{actual_cutoffs=}")
    
    # check forecast-points count per series
    cutoffs_by_series = forecasts.groupby(['unique_id', 'cutoff']).size().unstack('unique_id')
    pd.testing.assert_series_equal(cutoffs_by_series['Airline1'], cutoffs_by_series['Airline2'], check_names=False)
```


CopyAsk AI```
# Test predict_insample

def get_expected_cols(model, level):
    # index columns
    n_cols = 4
    for model in models:
        if isinstance(loss, (DistributionLoss, PMM, GMM, NBMM)):
            if level is None:
                # Variations of DistributionLoss return the sample mean as well
                n_cols += len(loss.quantiles) + 1
            else:
                # Variations of DistributionLoss return the sample mean as well
                n_cols += 2 * len(level) + 1
        else:
            if level is None:
                # Other probabilistic models return the sample mean as well
                n_cols += 1
            # Other probabilistic models return just the levels
            else:
                n_cols += len(level) + 1
    return n_cols

for loss in [
    # IQLoss(), 
    DistributionLoss(distribution="Normal", level=[80]),
    PMM(level=[80]),
]:
    for level in [None, [80, 90]]:
        models = [
            NHITS(h=12, input_size=12, loss=loss, max_steps=1),
            LSTM(h=12, input_size=12, loss=loss, max_steps=1, recurrent=True),
        ]
        nf = NeuralForecast(models=models, freq='D')

        nf.fit(df=AirPassengersPanel_train)
        df = nf.predict_insample(step_size=1, level=level)
        expected_cols = get_expected_cols(models, level)
        assert df.shape[1] == expected_cols, f'Shape mismatch for {loss} and level={level} in predict_insample: cols={df.shape[1]}, expected_cols={expected_cols}'
```


CopyAsk AI```
def config_optuna(trial):
    return {"input_size": trial.suggest_categorical('input_size', [12, 24]),
        "hist_exog_list": trial.suggest_categorical('hist_exog_list', [['trend'], ['y_[lag12]'], ['trend', 'y_[lag12]']]),
        "futr_exog_list": ['trend'],
        "max_steps": 10,
        "val_check_steps": 5}

config_ray = {'input_size': tune.choice([12, 24]), 
          'hist_exog_list': tune.choice([['trend'], ['y_[lag12]'], ['trend', 'y_[lag12]']]),
          'futr_exog_list': ['trend'],
          'max_steps': 10,
          'val_check_steps': 5}
```


CopyAsk AI```
# Test predict_insample step_size

h = 12
train_end = AirPassengers_pl['time'].max()
sizes = AirPassengers_pl['uid'].value_counts().to_numpy()

for step_size, test_size in [(7, 0), (9, 0), (7, 5), (9, 5)]:
    models = [NHITS(h=h, input_size=12, max_steps=1)]
    nf = NeuralForecast(models=models, freq='1mo')
    nf.fit(
        AirPassengers_pl,
        id_col='uid',
        time_col='time',
        target_col='target',    
    )
    # Note: only apply set_test_size() upon nf.fit(), otherwise it would have set the test_size = 0
    nf.models[0].set_test_size(test_size)    
    
    forecasts = nf.predict_insample(step_size=step_size)
    n_expected_cutoffs = (sizes[0][1] - test_size - nf.h + step_size) // step_size

    # compare cutoff values
    last_cutoff = train_end - test_size * pd.offsets.MonthEnd() - h * pd.offsets.MonthEnd()
    expected_cutoffs = np.flip(np.array([last_cutoff - step_size * i * pd.offsets.MonthEnd() for i in range(n_expected_cutoffs)]))
    pl_cutoffs = forecasts.filter(polars.col('uid') ==nf.uids[1]).select('cutoff').unique(maintain_order=True)
    actual_cutoffs = np.sort(np.array([pd.Timestamp(x['cutoff']) for x in pl_cutoffs.rows(named=True)]))
    np.testing.assert_array_equal(expected_cutoffs, actual_cutoffs, err_msg=f"{step_size=},{expected_cutoffs=},{actual_cutoffs=}")

    # check forecast-points count per series
    cutoffs_by_series = forecasts.group_by(['uid', 'cutoff']).count()
    assert_frame_equal(cutoffs_by_series.filter(polars.col('uid') == "Airline1").select(['cutoff', 'count']), cutoffs_by_series.filter(polars.col('uid') == "Airline2").select(['cutoff', 'count'] ), check_row_order=False)
```

