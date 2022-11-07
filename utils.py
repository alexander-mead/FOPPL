import numpy as np
import torch as tc
import wandb

def softplus(x, beta=1., threshold=20.):
    '''
    Original softplus: S(x) = (1/beta)*ln(1+exp(beta*x))
    '''
    s = tc.where(x<=threshold, tc.log(tc.exp(beta*x)+1.)/beta, x)
    return s


def inverse_softplus(s, beta=1., threshold=20.):
    '''
    Inverse softplus: x(S) = (1/beta)*ln(-1+exp(beta*S))
    '''
    x = tc.where(s<=threshold, tc.log(tc.exp(beta*s)-1.)/beta, s)
    return x


def covariance(x, y):
    '''
    Calculate the covariance between two sets of samples: x and y
    Assumes that rows correspond to independent measurements in each of x and y
    '''
    return (x*y).mean(axis=0)-x.mean(axis=0)*y.mean(axis=0)


def flatten_sample(sample):
    if type(sample) is list: # NOTE: Nasty hack for the output from program 4 of homework 2/4
        flat_sample = tc.concat([element.flatten() for element in sample])
    else:
        flat_sample = sample
    return flat_sample


def create_unique_list(list_with_duplicates):
    '''
    Takes a list that may contain duplicates and returns a new list with the duplicates removed
    '''
    #return list(set(list_with_duplicates)) # NOTE: This does not preserve order
    return list(dict.fromkeys(list_with_duplicates))


def burn_chain(samples, weights, burn_frac=None):
    '''
    Remove a certain fraction of points from the beginning of an MCMC chain
    '''
    if burn_frac is not None:
        n = len(samples)
        nburn = int(burn_frac*n)
        burned_samples = samples[nburn:]; burned_weights = weights[nburn:]
    return burned_samples, burned_weights


def log_sample(sample, i: int, wandb_name: str, resample=False) -> None:
    '''
    Log a single W&B sample
    '''
    #if sample.dtype is tc.bool: sample = sample.float() # NOTE: Hack to convert boolean to float H3Q3
    wandb_name_here = wandb_name+' samples' if not resample else wandb_name+' resamples'
    if sample.dim() == 0:
        samples_dict = {wandb_name_here+'; epoch': i, wandb_name_here: sample}
    else:
        samples_dict = {wandb_name_here+'; epoch': i, wandb_name_here: sample}
        for i, element in enumerate(sample):
            samples_dict[wandb_name_here+'; '+str(i)] = element
    wandb.log(samples_dict)


def log_params(variationals: list, i: int, wandb_name: str) -> None: 
    '''
    Log a set of variational-distribution parameters to W&B
    @params
        variationals: list of distributions corresponding to each sample node at each training step
        i: integer corresponding to epcoh
        wandb_name: string name of W&B run
    '''
    wandb_name_here = wandb_name+' params'
    samples_dict = {wandb_name_here+'; epoch': i}
    for node, distribution in variationals.items():
        params = [p.clone().detach().numpy() for p in distribution.params()]
        for i, param in enumerate(params):
            samples_dict[wandb_name_here+'; '+node+'; '+str(i)] = param
    wandb.log(samples_dict)


# def log_loss(losses: dict, i: int, wandb_name: str) -> None:
#     '''
#     Log a set of losses corresponding to each node to W&B
#     @params
#         losses: list of losses corresponding to each sample node at each training step
#         i: integer corresponding to epcoh
#         wandb_name: string name of W&B run
#     '''
#     wandb_name_here = wandb_name+' loss'
#     wandb_dict = {wandb_name_here+'; epoch': i}
#     for node, loss in losses.items():
#        wandb_dict[wandb_name_here+'; '+node] = loss
#     wandb.log(wandb_dict)
def log_loss(loss: float, i: int, wandb_name: str) -> None:
    '''
    Log a set of losses corresponding to each node to W&B
    @params
        losses: list of losses corresponding to each sample node at each training step
        i: integer corresponding to epcoh
        wandb_name: string name of W&B run
    '''
    wandb_name_here = wandb_name+' loss'
    wandb_dict = {
        wandb_name_here: loss,
        wandb_name_here+'; epoch': i,
    }
    wandb.log(wandb_dict)


def wandb_plots_homework2(samples, program):
    '''
    W&B logging of plots for homework 2
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'slope', 'bias'])
        wandb_log['Program 2; slope'] = wandb.plot.histogram(table, value='slope', title='Program 2; slope')
        wandb_log['Program 2; bias'] = wandb.plot.histogram(table, value='bias', title='Program 2; bias')
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        data = np.array(samples)
        xs = np.linspace(0, data.shape[1]-1, num=data.shape[1]) # [0, 1, ..., 16]
        x = []; y = []
        for i in range(data.shape[0]): # 1000 values
            for j in range(data.shape[1]): # 16 values
                x.append(xs[j])
                y.append(data[i, j])
        xedges = np.linspace(-0.5, data.shape[1]-0.5, data.shape[1]+1) # -0.5, 0.5, ..., 16.5
        yedges = np.linspace(-0.5, data.max()+0.5, int(data.max())+2) # -0.5, 0.5, 1.5, 2.5
        matrix, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[:-1]+0.5; ylabels = yedges[:-1]+0.5 # [0, 1, ..., 16]; [0, 1, 2]
        wandb_log['Program 3; heatmap'] = wandb.plots.HeatMap(xlabels, ylabels, matrix.T, show_text=True)
    elif program == 4:
        x_values = np.arange(samples.shape[1]+1)
        for y_values, name in zip([samples.mean(axis=0), samples.std(axis=0)], ['mean', 'std']):
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns=['position', name])
            title = 'Program 4; '+name
            wandb_log[title] = wandb.plot.line(table, 'position', name, title=title)
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)


def wandb_plots_homework3(samples, program):
    '''
    W&B logging of plots for homework 3
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'slope', 'bias'])
        wandb_log['Program 2; slope'] = wandb.plot.histogram(table, value='slope', title='Program 2; slope')
        wandb_log['Program 2; bias'] = wandb.plot.histogram(table, value='bias', title='Program 2; bias')
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x'])
        wandb_log['Program 3'] = wandb.plot.histogram(table, value='x', title='Program 3; Are the points from the same cluster?')
    elif program == 4:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x'])
        wandb_log['Program 4'] = wandb.plot.histogram(table, value='x', title='Program 4; Is it raining?')
    elif program == 5:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x', 'y'])
        wandb_log['Program 5; x'] = wandb.plot.histogram(table, value='x', title='Program 5; x')
        wandb_log['Program 5; y'] = wandb.plot.histogram(table, value='y', title='Program 5; y')
        wandb_log['Program 5; scatter'] = wandb.plot.scatter(table, x='x', y='y', title='Program 5; x vs. y')
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)


def wandb_plots_homework4(samples, program):
    '''
    W&B logging of plots for homework 4
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'slope', 'bias', 'posterior-predictive'])
        for thing in ['slope', 'bias', 'posterior-predictive']:
            wandb_log['Program 2; '+thing] = wandb.plot.histogram(table, value=thing, title='Program 2; '+thing)
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x'])
        wandb_log['Program 3'] = wandb.plot.histogram(table, value='x', title='Program 3; Are the points from the same cluster?')
    elif program == 4:
        data = np.array(samples)
        xs = np.linspace(0, data.shape[1]-1, num=data.shape[1]) # [0, 1, ..., 129]
        x = []; y = []
        for i in range(data.shape[0]): # Number of samples
            for j in range(data.shape[1]): # 130 values
                x.append(xs[j])
                y.append(data[i, j])
        xedges = np.linspace(-0.5, data.shape[1]-0.5, data.shape[1]+1) # -0.5, 0.5, ..., 129.5
        yedges = np.linspace(-10.5, 10.5, 20+2) # -10, -9, ..., 10
        matrix, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[:-1]+0.5; ylabels = yedges[:-1]+0.5
        wandb_log['Program 4; heatmap'] = wandb.plots.HeatMap(xlabels, ylabels, matrix.T)
    elif program == 5:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 's'])
        wandb_log['Program 5'] = wandb.plot.histogram(table, value='s', title='Program 5; s')
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)