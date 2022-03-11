#!/usr/bin/env python
import logging
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd
import json
import numpy as np
import time
import datetime as dt
from typing import Optional, Union

logger = logging.getLogger(__name__)


async def _download_single_ticker_info(session: aiohttp.ClientSession, ticker: str):
    """
    Download historical data for a single ticker with multithreading. Plus, it scrapes missing stock information.

    Parameters
    ----------
    ticker: str
        Ticker for which to download historical information.
    interval: str
        Frequency between data.
    start: str
        Start download data from this date.
    end: str
        End download data at this date.
    """

    
    try:
        async with session.get("https://finance.yahoo.com/quote/" + ticker) as response:
            html = await response.text()
        json_str = html.split("root.App.main =")[1].split("(this)")[0].split(";\n}")[0].strip()
        info = json.loads(json_str)["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["summaryProfile"]
        assert (len(info["sector"]) > 0) and (len(info["industry"]) > 0)
        return {"SYMBOL": ticker, "SECTOR": info["sector"], "INDUSTRY": info["industry"]}
    except Exception as e:
        return {"SYMBOL": ticker, "error": e}

        



async def _download_single_ticker_chart_data(
    session: aiohttp.ClientSession, ticker: str, start: int, end: int, interval: str = "1d"
) -> dict:
    """
    Download historical data for a single ticker.

    Parameters
    ----------
    ticker: str
        Ticker for which to download historical information.
    start: int
        Start download data from this timestamp date.
    end: int
        End download data at this timestamp date.
    interval: str
        Frequency between data.

    Returns
    -------
    data: dict
        Scraped dictionary of information.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = dict(period1=start, period2=end, interval=interval.lower(), includePrePost="false")

    async with session.get(url, params=params) as response:
        data_text = await response.text()

        if "Will be right back" in data_text:
            raise RuntimeError("*** YAHOO! FINANCE is currently down! ***\n")
        else:
            response_json = await response.json()
    try:
        raw_quotes = response_json["chart"]["result"][0]
        currency = raw_quotes["meta"]["currency"]
        return {"ticker": ticker, "quotes": _parse_quotes(raw_quotes), "currency": currency}
    except Exception as e:
        if "error" in response_json.get("chart", {}):
            e = response_json["chart"]["error"]
        return {"ticker": ticker, "error": e}

async def async_download_tickers_info(
    tickers: list, tickers_missing_info: list, start: int, end: int, interval: str,
):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        print("\nDownloading stock industry and sector")
        with logging_redirect_tqdm():
            tickers_info = await tqdm_asyncio.gather(
                *[_download_single_ticker_info(session, ticker) for ticker in tickers_missing_info]
            )

    no_error_tickers_info = [ticker_info for ticker_info in tickers_info if "error" not in ticker_info]
    errored_tickers_info = [ticker_info["SYMBOL"] for ticker_info in tickers_info if "error" in ticker_info]
    print(f"Out of {len(tickers_missing_info)} tickers missing info, we could get {len(no_error_tickers_info)}")
    print(f"Couldn't get info: {', '.join(errored_tickers_info)}")


    async with aiohttp.ClientSession(headers=headers) as session:
        print("\nDownloading stock quotes")
        with logging_redirect_tqdm():
            all_tickers_chart_data = await tqdm_asyncio.gather(
                *[_download_single_ticker_chart_data(session, ticker, start, end, interval) for ticker in tickers]
            )
    quotes = {ticker_dict["ticker"]: ticker_dict["quotes"] for ticker_dict in all_tickers_chart_data if "error" not in ticker_dict}
    errored_tickers_quotes = [ticker_dict["ticker"] for ticker_dict in all_tickers_chart_data if "error" in ticker_dict]
    print(f"Out of {len(tickers)} tickers, we could get quotes for {len(quotes)}")
    print(f"Couldn't get quotes for: {', '.join(errored_tickers_quotes)}")
    quotes_df = pd.concat(quotes.values(), keys=quotes.keys(), axis=1, sort=True)

    stock_info_df = pd.DataFrame(no_error_tickers_info, columns=["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"])
    currencies = {ticker_dict["ticker"]: ticker_dict["currency"] for ticker_dict in all_tickers_chart_data if "error" not in ticker_dict}
    stock_info_df["CURRENCY"] = stock_info_df["SYMBOL"].apply(currencies.get)

    return quotes_df, stock_info_df


def download_tickers_info(
    tickers: list, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d"
) -> dict:
    """
    Download historical data for tickers in the list.

    Parameters
    ----------
    tickers: list
        Tickers for which to download historical information.
    start: str or int
        Start download data from this date.
    end: str or int
        End download data at this date.
    interval: str
        Frequency between data.

    Returns
    -------
    data: dict
        Dictionary including the following keys:
        - tickers: list of tickers
        - logp: array of log-adjusted closing prices, shape=(num stocks, length period);
        - volume: array of volumes, shape=(num stocks, length period);
        - sectors: dictionary of stock sector for each ticker;
        - industries: dictionary of stock industry for each ticker.
    """
    tickers = tickers if isinstance(tickers, (list, set, tuple)) else tickers.replace(",", " ").split()
    tickers = list(set([ticker.upper() for ticker in tickers]))

    logger.info(f"Downloading data for {len(tickers)} tickers")

    stock_info_columns = ["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"]
    stock_info_filename = "stock_info.csv"

    try:
        cached_stock_info_df = pd.read_csv(stock_info_filename)
        logger.info(f"Reading stock info found in file '{stock_info_filename}'")
    except FileNotFoundError:
        cached_stock_info_df = pd.DataFrame(columns=stock_info_columns)

    tickers_already_fetched_info = cached_stock_info_df["SYMBOL"].values

    tickers_misssing_info = [ticker for ticker in tickers if ticker not in tickers_already_fetched_info]

    end = int(time.time()) if end is None else int(dt.datetime.strptime(end, "%Y-%m-%d").timestamp())
    start = (
        int((dt.datetime.today() - dt.timedelta(365)).timestamp())
        if start is None
        else int(dt.datetime.strptime(start, "%Y-%m-%d").timestamp())
    )

    stocks_quotes_df, newly_fetched_stock_info_df = asyncio.run(
        async_download_tickers_info(tickers, tickers_misssing_info, start, end, interval)
    )

    full_stock_info_df = pd.concat([cached_stock_info_df, newly_fetched_stock_info_df])
    full_stock_info_df.to_csv(stock_info_filename, index=False)

    if stocks_quotes_df.shape[0] == 0:
        raise Exception("No symbol with full information is available.")

    stocks_quotes_df = stocks_quotes_df.drop(
        columns=stocks_quotes_df.columns[stocks_quotes_df.isnull().sum(0) > 0.33 * stocks_quotes_df.shape[0]]
    )
    stocks_quotes_df = stocks_quotes_df.fillna(method="bfill").fillna(method="ffill").drop_duplicates()

    tickers_misssing_info = [
        ticker for ticker in tickers if ticker not in stocks_quotes_df.columns.get_level_values(0)[::2].tolist()
    ]
    tickers = stocks_quotes_df.columns.get_level_values(0)[::2].tolist()
    if len(tickers_misssing_info) > 0:
        print(
            f"\nRemoving {tickers_misssing_info} from list of symbols because we could not collect full information."
        )

    # download exchange rates and convert to most common currency
    currencies = full_stock_info_df["CURRENCY"].to_list()
    default_currency = full_stock_info_df["CURRENCY"].mode()[0]
    xrates = get_exchange_rates(currencies, default_currency, stocks_quotes_df.index, start, end, interval)

    return dict(
        tickers=tickers,
        dates=pd.to_datetime(stocks_quotes_df.index),
        price=stocks_quotes_df.iloc[:, stocks_quotes_df.columns.get_level_values(1) == "Adj Close"].to_numpy().T,
        volume=stocks_quotes_df.iloc[:, stocks_quotes_df.columns.get_level_values(1) == "Volume"].to_numpy().T,
        currencies=currencies,
        exchange_rates=xrates,
        default_currency=default_currency,
        sectors={
            stock_info_row["SYMBOL"]: stock_info_row["SECTOR"] for _, stock_info_row in full_stock_info_df.iterrows()
        },
        industries={
            stock_info_row["SYMBOL"]: stock_info_row["INDUSTRY"] for _, stock_info_row in full_stock_info_df.iterrows()
        },
    )


def _parse_quotes(data: dict, parse_volume: bool = True) -> pd.DataFrame:
    """
    It creates a data frame of adjusted closing prices, and, if `parse_volume=True`, volumes. If no adjusted closing
    price is available, it sets it equal to closing price.

    Parameters
    ----------
    data: dict
        Data containing historical information of corresponding stock.
    parse_volume: bool
        Include or not volume information in the data frame.
    """
    timestamps = data["timestamp"]
    ohlc = data["indicators"]["quote"][0]
    closes = ohlc["close"]
    if parse_volume:
        volumes = ohlc["volume"]
    try:
        adjclose = data["indicators"]["adjclose"][0]["adjclose"]
    except:
        adjclose = closes

    # fix NaNs in the second-last entry of adjusted closing prices
    if adjclose[-2] is None:
        adjclose[-2] = adjclose[-1]

    assert (np.array(adjclose) > 0).all()

    quotes = {"Adj Close": adjclose}
    if parse_volume:
        quotes["Volume"] = volumes
    quotes = pd.DataFrame(quotes)
    quotes.index = pd.to_datetime(timestamps, unit="s").date
    quotes.sort_index(inplace=True)
    quotes = quotes.loc[~quotes.index.duplicated(keep="first")]

    return quotes


def get_exchange_rates(
    from_currencies: list,
    to_currency: str,
    dates: pd.Index,
    start: Union[str, int] = None,
    end: Union[str, int] = None,
    interval: str = "1d",
) -> dict:
    """
    It finds the most common currency and set it as default one. For any other currency, it downloads exchange rate
    closing prices to the default currency and return them as data frame.

    Parameters
    ----------
    from_currencies: list
        A list of currencies to convert.
    to_currency: str
        Currency to convert to.
    dates: date
        Dates for which exchange rates should be available.
    start: str or int
        Start download data from this timestamp date.
    end: str or int
        End download data at this timestamp date.
    interval: str
        Frequency between data.

    Returns
    -------
    xrates: dict
        A dictionary with currencies as keys and list of exchange rates at desired dates as values.
    """
    if end is None:
        end = int(dt.datetime.timestamp(dt.datetime.today()))
    elif type(end) is str:
        end = int(dt.datetime.timestamp(dt.datetime.strptime(end, "%Y-%m-%d")))
    if start is None:
        start = int(dt.datetime.timestamp(dt.datetime.today() - dt.timedelta(365)))
    elif type(start) is str:
        start = int(dt.datetime.timestamp(dt.datetime.strptime(start, "%Y-%m-%d")))

    ucurrencies, counts = np.unique(from_currencies, return_counts=True)
    tmp = {}
    if to_currency not in ucurrencies or len(ucurrencies) > 1:
        for curr in ucurrencies:
            if curr != to_currency:
                tmp[curr] = _download_single_ticker_chart_data(curr + to_currency + "=x", start, end, interval)
                tmp[curr] = _parse_quotes(tmp[curr]["chart"]["result"][0], parse_volume=False)["Adj Close"]
        tmp = pd.concat(tmp.values(), keys=tmp.keys(), axis=1, sort=True)
        xrates = pd.DataFrame(index=dates, columns=tmp.columns)
        xrates.loc[xrates.index.isin(tmp.index)] = tmp
        xrates = xrates.fillna(method="bfill").fillna(method="ffill")
        xrates.to_dict(orient="list")
    else:
        xrates = tmp
    return xrates
