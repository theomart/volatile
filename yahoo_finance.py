import json
from typing import Dict, Optional
import aiohttp
import numpy as np

import pandas as pd
import logging


logger = logging.getLogger(__name__)


async def download_ticker_sector_industry(session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
    """
    Download historical data for a single ticker with multithreading. Plus, it scrapes missing stock information.

    Parameters
    ----------
    ticker: str
        Ticker for which to download historical information.
    """

    try:
        async with session.get("https://finance.yahoo.com/quote/" + ticker) as response:
            html = await response.text()
        json_str = html.split("root.App.main =")[1].split("(this)")[0].split(";\n}")[0].strip()
        info = json.loads(json_str)["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["summaryProfile"]
        assert (len(info["sector"]) > 0) and (len(info["industry"]) > 0)
        return {"SYMBOL": ticker, "SECTOR": info["sector"], "INDUSTRY": info["industry"]}
    except Exception as e:
        logger.warning(f"Error downloading info for {ticker=}: {e}")
        return None


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
        return {
            "ticker": ticker,
            "quotes": _parse_quotes(response_json),
            "currency": extract_currency_from_chart_json(response_json),
        }
    except Exception as e:
        if "error" in response_json.get("chart", {}):
            e = response_json["chart"]["error"]
            logger.warning(f"Downloading chart data for ticker {ticker} threw error")
        return None


def extract_currency_from_chart_json(response_json):
    return response_json["chart"]["result"][0]["meta"]["currency"]


def _parse_quotes(response_json: dict) -> pd.DataFrame:
    """
    It creates a data frame of adjusted closing prices and volumes. If no adjusted closing
    price is available, it sets it equal to closing price.

    Parameters
    ----------
    data: dict
        Data containing historical information of corresponding stock.
    """

    data = response_json["chart"]["result"][0]

    quotes = {}
    timestamps = data["timestamp"]
    indicators = data["indicators"]
    ohlc = indicators["quote"][0]
    closes = ohlc["close"]

    quotes["Volume"] = ohlc["volume"]

    try:
        adjclose = indicators["adjclose"][0]["adjclose"]
    except (KeyError, IndexError):
        adjclose = closes

    # fix NaNs in the second-last entry of adjusted closing prices
    if adjclose[-2] is None:
        adjclose[-2] = adjclose[-1]

    assert (np.array(adjclose) > 0).all()

    quotes["Adj Close"] = adjclose

    quotes = pd.DataFrame(quotes)
    quotes.index = pd.to_datetime(timestamps, unit="s").date
    quotes.sort_index(inplace=True)
    quotes = quotes.loc[~quotes.index.duplicated(keep="first")]

    return quotes
