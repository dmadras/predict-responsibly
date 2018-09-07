import os

def get_default_kwargs(data_dir, opts, model_opts):
    """some defaults kwargs dicts for declaring models and datasets, e.g.,

    >>> data_kwargs, model_kwargs = get_default_kwargs('compas', args)
    >>> data = MyDataset(**data_kwargs)
    >>> model = MyModel(**model_kwargs)

    """
    dataset_agnostic_data_kwargs = dict(
        seed=opts['data_random_seed'], 
        use_attr=opts['use_attr']
    )
    dataset_kwargs = dict(
        name=model_opts['name'],
        attr0_name=model_opts['attr0_name'],
        attr1_name=model_opts['attr1_name'],
        npzfile=os.path.join(data_dir, model_opts['npzfiles'][opts['dm_type']]),
        use_attr=model_opts['use_attr']
    )
    
    model_kwargs = dict(
        seed=opts['model_random_seed'], 
        pass_coeff=opts['pass_coeff'],
        fair_coeff=opts['fair_coeff'],
        adim=1,
        ydim=1,
        xdim=model_opts['xdim'] + (1 if model_opts['use_attr'] else 0),
        hidden_layer_specs=model_opts['hidden_layer_specs']
    )

    return {**dataset_agnostic_data_kwargs, **dataset_kwargs}, model_kwargs
