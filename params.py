import sys


def algo_params(param_str):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if param_str == 'local_search': 
        params.append({'algo_name':'local_search', 'total_queries':300})

    elif param_str == 'ls_cont_at_min':
        params.append({'algo_name':'local_search', 'total_queries':300, 'stop_at_minimum':False})

    elif param_str == 'ls_query_part':
        params.append({'algo_name':'local_search', 'total_queries':300, 'query_full_nbhd':False})

    elif param_str == 'test':
        params.append({'algo_name':'random', 'total_queries':30})
        params.append({'algo_name':'evolution', 'total_queries':30})
        params.append({'algo_name':'bananas', 'total_queries':30})   
        params.append({'algo_name':'gp_bayesopt', 'total_queries':30})
        params.append({'algo_name':'dngo', 'total_queries':30})

    elif param_str == 'test_simple': 
        params.append({'algo_name':'random', 'total_queries':30})
        params.append({'algo_name':'evolution', 'total_queries':30})

    elif param_str == 'main_experiments':
        params.append({'algo_name':'random', 'total_queries':300})
        params.append({'algo_name':'evolution', 'total_queries':300})
        params.append({'algo_name':'bananas', 'total_queries':300})  
        params.append({'algo_name':'gp_bayesopt', 'total_queries':300})        
        params.append({'algo_name':'dngo', 'total_queries':300})
        params.append({'algo_name':'local_search', 'total_queries':300, 'stop_at_minimum':False})
        params.append({'algo_name':'local_search', 'total_queries':300, 'query_full_nbhd':False})

    elif param_str == 'bananas':
        params.append({'algo_name':'bananas', 'total_queries':150})

    else:
        print('invalid algorithm params: {}'.format(param_str))
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params


def meta_neuralnet_params(param_str):

    if param_str == 'nasbench':
        params = {'search_space':'nasbench', 'dataset':'cifar10', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'darts':
        params = {'search_space':'darts', 'dataset':'cifar10', 'loss':'mape', 'num_layers':10, 'layer_width':20, \
            'epochs':10000, 'batch_size':32, 'lr':.00001, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_201_cifar10':
        params = {'search_space':'nasbench_201', 'dataset':'cifar10', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_201_cifar100':
        params = {'search_space':'nasbench_201', 'dataset':'cifar100', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'nasbench_201_imagenet':
        params = {'search_space':'nasbench_201', 'dataset':'ImageNet16-120', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    else:
        print('invalid meta neural net params: {}'.format(param_str))
        sys.exit()

    return params
