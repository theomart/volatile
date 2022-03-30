#!/usr/bin/env python
import logging
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd
import numpy as np
import time
import datetime as dt
from typing import Collection, Dict, List, Optional, Tuple, Union

from yahoo_finance import _download_single_ticker_chart_data, download_ticker_sector_industry

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
}


async def download_tickers_sector_industry(tickers: List[str]) -> pd.DataFrame:
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        print("\nDownloading stock industry and sector")
        with logging_redirect_tqdm():
            tickers_info = await tqdm_asyncio.gather(
                *[download_ticker_sector_industry(session, ticker) for ticker in tickers]
            )

    if None in tickers_info:
        errored_tickers = [ticker for ticker, ticker_info in zip(tickers, tickers_info) if ticker_info is None]
        tickers_info = [ticker_info for ticker_info in tickers_info if ticker_info is not None]
        print(f"Out of {len(tickers)} tickers missing info, we could get {len(tickers_info)}")
        print(f"Couldn't get info for the following {len(errored_tickers)}: {', '.join(errored_tickers)}")

    return pd.DataFrame(tickers_info, columns=["SYMBOL", "SECTOR", "INDUSTRY"])


async def download_tickers_quotes(
    tickers: List[str], start_date: int, end_date: int, interval: str
) -> Tuple[pd.DataFrame, Dict]:
    """Download quotes and their currencies for all the specified tickers in the specified time window.

    Parameters
    ----------
    tickers : List[str]
        The list of tickers to download data for
    start_date : int
        The start date in POSIX format.
    end_date : int
        The end date in POSIX format.
    interval : str
        The interval between each data point (e.g. "1d")

    Returns
    -------
    Tuple[List[Dict], Dict]
        A tuple containg two dicts, first the quotes, second their currencies.
    """
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        print("\nDownloading stock quotes")
        with logging_redirect_tqdm():
            tickers_chart_data = await tqdm_asyncio.gather(
                *[
                    _download_single_ticker_chart_data(session, ticker, start_date, end_date, interval)
                    for ticker in tickers
                ]
            )

    if None in tickers_chart_data:
        errored_tickers = [ticker for ticker, ticker_info in zip(tickers, tickers_chart_data) if ticker_info is None]
        tickers_chart_data = [t for t in tickers_chart_data if t is not None]
        print(f"Out of {len(tickers)} tickers, we could get quotes for {len(tickers_chart_data)}")
        print(f"Couldn't get quotes for: {', '.join(errored_tickers)}")

    quotes = {ticker_dict["ticker"]: ticker_dict["quotes"] for ticker_dict in tickers_chart_data}
    currencies = {ticker_dict["ticker"]: ticker_dict["currency"] for ticker_dict in tickers_chart_data}

    return pd.concat(quotes, axis="columns", sort=True), currencies


def extract_ticker_list(tickers: Union[Collection[str], str]) -> List[str]:
    if isinstance(tickers, (list, set, tuple)):
        pass
    elif isinstance(tickers, str):
        # Replacing commas by spaces helps removing excess spaces between commas if any
        tickers = tickers.replace(",", " ").split()
    else:
        raise ValueError("tickers must be a str consisting of a comma separated list of tickers or a list of tickers")

    return list(set([ticker.upper() for ticker in tickers]))


def parse_start_end_date(
    start_date: Optional[str] = None, end_date: Optional[str] = None, default_start_days_ago=365
) -> Tuple[int, int]:
    end_date = int(time.time()) if end_date is None else int(dt.datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    start_date = (
        int((dt.datetime.today() - dt.timedelta(365)).timestamp())
        if start_date is None
        else int(dt.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    )
    return start_date, end_date


def download_tickers_info(
    tickers: list, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: str = "1d"
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

    logger.info(f"Downloading data for {len(tickers)} tickers")

    tickers = extract_ticker_list(tickers)

    stock_info_filename = "stock_info.csv"

    try:
        stock_info_df = pd.read_csv(stock_info_filename)
        logger.info(f"Reading stock info found in file '{stock_info_filename}'")
    except FileNotFoundError:
        # Creating an empty dataframe
        stock_info_columns = ["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"]
        stock_info_df = pd.DataFrame(columns=stock_info_columns)

    # Downloading stock quotes and currencies
    start_date, end_date = parse_start_end_date(start_date, end_date)
    stocks_quotes_df, currencies = asyncio.run(download_tickers_quotes(tickers, start_date, end_date, interval))

    # Remove tickers with excess null values
    stocks_quotes_df = stocks_quotes_df.loc[:, (stocks_quotes_df.isnull().mean() < 0.33)]
    assert stocks_quotes_df.shape[0] > 0, Exception("No symbol with full information is available.")

    # Fill in null values
    stocks_quotes_df = stocks_quotes_df.fillna(method="bfill").fillna(method="ffill").drop_duplicates()

    final_list_tickers = stocks_quotes_df.columns.get_level_values(0).unique()
    failed_to_get_tickers_quotes = [ticker for ticker in tickers if ticker not in final_list_tickers]
    if len(failed_to_get_tickers_quotes) > 0:
        print(
            f"\nRemoving {failed_to_get_tickers_quotes} from list of symbols because we could not collect complete quotes."
        )

    # Downloading missing stocks info
    tickers_already_fetched_info = stock_info_df["SYMBOL"].values
    tickers_missing_info = [ticker for ticker in tickers if ticker not in tickers_already_fetched_info]
    if len(tickers_missing_info) > 0:
        missing_tickers_info_df = asyncio.run(download_tickers_sector_industry(tickers_missing_info))
        missing_tickers_info_df["CURRENCY"] = missing_tickers_info_df["SYMBOL"].apply(currencies.get)
        stock_info_df = pd.concat([stock_info_df, missing_tickers_info_df])
        stock_info_df.to_csv(stock_info_filename, index=False)

    # Taking the quote currency as the one that appears the most in the data
    default_currency = stock_info_df["CURRENCY"].mode()[0]
    # Downloading the exchange rate between the default currency and all the others in the data
    currencies = stock_info_df["CURRENCY"].to_list()
    exchange_rates = get_exchange_rates(
        from_currencies=stock_info_df["CURRENCY"].dropna().to_list(),
        to_currency=default_currency,
        dates_index=stocks_quotes_df.index,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    return dict(
        tickers=final_list_tickers,
        dates=pd.to_datetime(stocks_quotes_df.index),
        price=stocks_quotes_df.xs("Adj Close", level=1, axis="columns").to_numpy().T,
        volume=stocks_quotes_df.xs("Volume", level=1, axis="columns").to_numpy().T,
        currencies=currencies,
        exchange_rates=exchange_rates,
        default_currency=default_currency,
        sectors={ticker: sector for ticker, sector in zip(stock_info_df["SYMBOL"], stock_info_df["SECTOR"])},
        industries={ticker: industry for ticker, industry in zip(stock_info_df["SYMBOL"], stock_info_df["INDUSTRY"])},
    )


def get_exchange_rates(
    from_currencies: list,
    to_currency: str,
    dates_index: pd.DatetimeIndex,
    start_date: int,
    end_date: int,
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

    from_currencies = [currency for currency in np.unique(from_currencies) if currency != to_currency]

    if len(from_currencies) == 0:
        return {}

    xrates = asyncio.run(async_get_exchange_rates(from_currencies, to_currency, start_date, end_date, interval))
    xrates.reindex = dates_index
    xrates = xrates.fillna(method="bfill").fillna(method="ffill")

    return xrates.to_dict(orient="list")


async def async_get_exchange_rates(
    from_currencies: list,
    to_currency: str,
    start_date: int,
    end_date: int,
    interval: str,
):
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        currencies_chart_data = await asyncio.gather(
            *[
                _download_single_ticker_chart_data(
                    session, from_currency + to_currency + "=x", start_date, end_date, interval
                )
                for from_currency in from_currencies
            ]
        )

    quotes = [chart_data["quotes"]["Adj Close"] for chart_data in currencies_chart_data]
    return pd.concat(quotes, keys=from_currencies, axis="columns", sort=True)
