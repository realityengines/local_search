import sys


def algo_params(param_str):

    params = []

    if param_str == 'local_search':
        params.append({'algo_name':'local_search', 'total_queries':500, 'query_full_nbhd':True, 'stop_at_minimum':True, 'num_init':1})

    elif param_str == 'run_all':
        params.append({'algo_name':'random', 'total_queries':500})
        params.append({'algo_name':'evolution', 'total_queries':500})
        params.append({'algo_name':'local_search', 'total_queries':500})        


    else:
        print('invalid algorithm params: {}'.format(param_str))
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params


def meta_neuralnet_params(param_str):

    if param_str == 'nasbench':
        params = {'search_space':'nasbench', 'dataset':'cifar10', 'mf':False, 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'darts':
        params = {'search_space':'darts', 'dataset':'cifar10', 'mf':False, 'loss':'mape', 'num_layers':10, 'layer_width':20, \
            'epochs':10000, 'batch_size':32, 'lr':.00001, 'regularization':0, 'verbose':0}

    else:
        print('invalid meta neural net params: {}'.format(param_str))
        sys.exit()

    return params
