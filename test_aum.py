import requests
from bs4 import BeautifulSoup
import re

url = "https://amc.ppfas.com/schemes/nav-history/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(url, headers=headers)
html = r.text

# Find Parag Parikh Flexi Cap section + AUM
match = re.search(
    r'Parag Parikh Flexi Cap Fund.*?AUM.*?([\d,]+\.\d+)',
    html,
    re.S
)

if match:
    aum = match.group(1)
    print("Parag Parikh Flexi Cap Fund AUM =", aum, "Cr")
else:
    print("AUM not found")