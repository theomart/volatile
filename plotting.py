import os
import numpy as np
import matplotlib.pyplot as plt

def compute_uncertainty_bounds(est: np.array, std: np.array):
    return np.maximum(0, est - 2 * std), est + 2 * std

def plot_market_estimates(data: dict, est: np.array, std: np.array):
    """
    It makes a market estimation plot with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    est: np.array
        Price trend estimate at market-level.
    std: np.array
        Standard deviation estimate of price trend at market-level.
    """
    print('\nPlotting market estimation...')
    fig = plt.figure(figsize=(10, 3))
    logp = np.log(data['price'])
    t = logp.shape[1]
    lb, ub = compute_uncertainty_bounds(est, std)

    plt.grid(axis='both')
    plt.title("Market", fontsize=15)
    avg_price = np.exp(logp.mean(0))
    l1 = plt.plot(data["dates"], avg_price, label="avg. price in {}".format(data['default_currency']), color="C0")
    l2 = plt.plot(data["dates"], est[0], label="trend", color="C1")
    l3 = plt.fill_between(data["dates"], lb[0], ub[0], alpha=0.2, label="+/- 2 st. dev.", color="C0")
    plt.ylabel("avg. price in {}".format(data['default_currency']), fontsize=12)
    plt.twinx()
    l4 = plt.bar(data["dates"], data['volume'].mean(0), width=1, color='g', alpha=0.2, label='avg. volume')
    l4[0].set_edgecolor('r')
    for d in range(1, t):
        if avg_price[d] - avg_price[d - 1] < 0:
            l4[d].set_color('r')
    plt.ylabel("avg. volume", fontsize=12)
    ll = l1 + l2 + [l3] + [l4]
    labels = [l.get_label() for l in ll]
    plt.legend(ll, labels, loc="upper left")
    fig_name = 'market_estimation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Market estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))

def plot_sector_estimates(data: dict, info: dict, est: np.array, std: np.array):
    """
    It makes a plot for each sector with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    info: dict
        Model hierarchy information.
    est: np.array
        Price trend estimate at sector-level.
    std: np.array
        Standard deviation estimate of price trend at sector-level.
    """
    print('\nPlotting sector estimation...')
    num_columns = 3
    logp = np.log(data['price'])
    t = logp.shape[1]
    lb, ub = compute_uncertainty_bounds(est, std)

    NA_sectors = np.where(np.array([sec[:2] for sec in info['unique_sectors']]) == "NA")[0]
    num_NA_sectors = len(NA_sectors)
    
    fig = plt.figure(figsize=(20, max(info['num_sectors'] - num_NA_sectors, 5)))
    j = 0
    for i in range(info['num_sectors']):
        if i not in NA_sectors:
            j += 1
            plt.subplot(int(np.ceil((info['num_sectors'] - num_NA_sectors) / num_columns)), num_columns, j)
            plt.grid(axis='both')
            plt.title(info['unique_sectors'][i], fontsize=15)
            idx_sectors = np.where(np.array(info['sectors_id']) == i)[0]
            avg_price = np.exp(logp[idx_sectors].reshape(-1, t).mean(0))
            l1 = plt.plot(data["dates"], avg_price,
                          label="avg. price in {}".format(data['default_currency']), color="C0")
            l2 = plt.plot(data["dates"], est[i], label="trend", color="C1")
            l3 = plt.fill_between(data["dates"], lb[i], ub[i], alpha=0.2, label="+/- 2 st. dev.",
                                  color="C0")
            plt.ylabel("avg. price in {}".format(data['default_currency']), fontsize=12)
            plt.xticks(rotation=45)
            plt.twinx()
            l4 = plt.bar(data["dates"],
                         data['volume'][np.where(np.array(info['sectors_id']) == i)[0]].reshape(-1, t).mean(0),
                         width=1, color='g', alpha=0.2, label='avg. volume')
            for d in range(1, t):
                if avg_price[d] - avg_price[d - 1] < 0:
                    l4[d].set_color('r')
            l4[0].set_edgecolor('r')
            plt.ylabel("avg. volume", fontsize=12)
            ll = l1 + l2 + [l3] + [l4]
            labels = [l.get_label() for l in ll]
            plt.legend(ll, labels, loc="upper left")
    
    plt.tight_layout()
    fig_name = 'sector_estimation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Sector estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))

def plot_industry_estimates(data: dict, info: dict, est: np.array, std: np.array):
    """
    It makes a plot for each industry with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    info: dict
        Model hierarchy information.
    est: np.array
        Price trend estimate at industry-level.
    std: np.array
        Standard deviation estimate of price trend at industry-level.
    """
    print('\nPlotting industry estimation...')
    num_columns = 3
    logp = np.log(data['price'])
    t = logp.shape[1]
    lb, ub = compute_uncertainty_bounds(est, std)

    NA_industries = np.where(np.array([ind[:2] for ind in info['unique_industries']]) == "NA")[0]
    num_NA_industries = len(NA_industries)

    fig = plt.figure(figsize=(20, max(info['num_industries'] - num_NA_industries, 5)))
    j = 0
    for i in range(info['num_industries']):
        if i not in NA_industries:
            j += 1
            plt.subplot(int(np.ceil((info['num_industries'] - num_NA_industries) / num_columns)), num_columns, j)
            plt.grid(axis='both')
            plt.title(info['unique_industries'][i], fontsize=15)
            idx_industries = np.where(np.array(info['industries_id']) == i)[0]
            plt.title(info['unique_industries'][i], fontsize=15)
            avg_price = np.exp(logp[idx_industries].reshape(-1, t).mean(0))
            l1 = plt.plot(data["dates"], avg_price,
                          label="avg. price in {}".format(data['default_currency']), color="C0")
            l2 = plt.plot(data["dates"], est[i], label="trend", color="C1")
            l3 = plt.fill_between(data["dates"], lb[i], ub[i], alpha=0.2, label="+/- 2 st. dev.",
                                  color="C0")
            plt.ylabel("avg. price in {}".format(data['default_currency']), fontsize=12)
            plt.xticks(rotation=45)
            plt.twinx()
            l4 = plt.bar(data["dates"],
                         data['volume'][np.where(np.array(info['industries_id']) == i)[0]].reshape(-1, t).mean(0),
                         width=1, color='g', alpha=0.2, label='avg. volume')
            for d in range(1, t):
                if avg_price[d] - avg_price[d - 1] < 0:
                    l4[d].set_color('r')
            l4[0].set_edgecolor('r')
            plt.ylabel("avg. volume", fontsize=12)
            ll = l1 + l2 + [l3] + [l4]
            labels = [l.get_label() for l in ll]
            plt.legend(ll, labels, loc="upper left")
    plt.tight_layout()
    fig_name = 'industry_estimation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Industry estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))

def plot_stock_estimates(data: dict, est: np.array, std: np.array, rank_type: str, rank: list, ranked_rates: np.array):
    """
    It makes a plot for each stock with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    est: np.array
        Price trend estimate at stock-level.
    std: np.array
        Standard deviation estimate of price trend at stock-level.
    rank_type: str
        Type of rank. It can be either `rate` or `growth`.
    rank: list
        List of integers at stock-level indicating the rank specified in `rank_type`.
    ranked_rates: np.array
        Array of rates at stock-level ranked according to `rank`.
    """
    num_stocks, t = data['price'].shape

    # determine which stocks are along trend to avoid plotting them
    if rank_type == "rate":
        to_plot = np.where(np.array(ranked_rates) != "ALONG TREND")[0]
    else:
        to_plot = np.where(np.array(ranked_rates) == "ALONG TREND")[0][:99]
    dont_plot = [x for x in np.arange(num_stocks) if x not in to_plot]
    num_to_plot = len(to_plot)
    if num_to_plot > 0:
        print('\nPlotting stock estimation...')
        num_columns = 3

        ranked_tickers = np.array(data['tickers'])[rank]
        ranked_p = data['price'][rank]
        ranked_volume = data['volume'][rank]
        ranked_currencies = np.array(data['currencies'])[rank]
        ranked_est = est[rank]
        ranked_std = std[rank]

        ranked_lb, ranked_ub = compute_uncertainty_bounds(ranked_est, ranked_std)

        j = 0
        fig = plt.figure(figsize=(20, max(num_to_plot, 5)))
        for i in range(num_stocks):
            if i not in dont_plot:
                j += 1
                plt.subplot(int(np.ceil(num_to_plot / num_columns)), num_columns, j)
                plt.grid(axis='both')
                plt.title(ranked_tickers[i], fontsize=15)
                l1 = plt.plot(data["dates"], ranked_p[i], label="price in {}".format(ranked_currencies[i]))
                l2 = plt.plot(data["dates"], ranked_est[i], label="trend")
                l3 = plt.fill_between(data["dates"], ranked_lb[i], ranked_ub[i], alpha=0.2,
                                      label="+/- 2 st. dev.")
                plt.yticks(fontsize=12)
                plt.xticks(rotation=45)
                plt.ylabel("price in {}".format(ranked_currencies[i]), fontsize=12)
                plt.twinx()
                l4 = plt.bar(data["dates"], ranked_volume[i], width=1, color='g', alpha=0.2, label='volume')
                for d in range(1, t):
                    if ranked_p[i, d] - ranked_p[i, d - 1] < 0:
                        l4[d].set_color('r')
                l4[0].set_edgecolor('r')
                plt.ylabel("volume", fontsize=12)
                ll = l1 + l2 + [l3] + [l4]
                labels = [l.get_label() for l in ll]
                plt.legend(ll, labels, loc="upper left")
        plt.tight_layout()
        fig_name = 'stock_estimation.png'
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Stock estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))

    elif os.path.exists('stock_estimation.png'):
        os.remove('stock_estimation.png')