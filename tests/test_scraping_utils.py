from bs4 import BeautifulSoup

from nfl_predictor.utils import scraping_utils


class DummyResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


def test_parse_tr_rating_table() -> None:
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
        <tr><td>1</td><td>Buffalo Bills (10-4)</td><td>6.7</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    teams, ratings = scraping_utils._parse_tr_rating_table(table)

    assert teams == ["BUF"]
    assert ratings == [6.7]


def test_parse_tr_stat_table() -> None:
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Kansas City Chiefs (11-3)</td><td>52.5%</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    teams, stats = scraping_utils._parse_tr_stat_table(table)

    assert teams == ["KC"]
    assert stats == [52.5]


def test_scrape_survivor_grid_spreads_parses(monkeypatch) -> None:
    rows = [
        """
        <tr>
            <td>BUF(10-4)</td>
            <td>@KC-3.5</td>
            <td>MIA+2</td>
        </tr>
        """
    ]
    rows.extend(
        f"<tr><td>XXX{i}</td><td>BYE</td><td>BYE</td></tr>" for i in range(30)
    )
    html = f"""
    <html>
        <body>
            <table>
                <tr><th>Team</th><th>16</th><th>17</th></tr>
                {''.join(rows)}
            </table>
        </body>
    </html>
    """

    def fake_get(*_args, **_kwargs):
        return DummyResponse(html)

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)

    spreads = scraping_utils.scrape_survivor_grid_spreads()
    assert spreads["BUF"][16] == -3.5
    assert spreads["BUF"][17] == 2.0
