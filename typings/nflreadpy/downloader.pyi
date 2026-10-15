import requests

class NflverseDownloader:
    session: requests.Session

def get_downloader() -> NflverseDownloader: ...
