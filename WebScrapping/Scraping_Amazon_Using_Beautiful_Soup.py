# Python program to scrape Amazon Smart Home page
import requests
from bs4 import BeautifulSoup
import csv

URL = "https://www.amazon.com/gp/browse.html?node=6563140011"

headers = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(URL, headers=headers)

soup = BeautifulSoup(r.content, 'html.parser')

quotes = []  # same variable name (not changed)

# Main container (Amazon uses divs, not id=all_items)
table = soup.find('div', attrs={'id': 'desktop-dp-sims_session-similarities'})

if table is None:
    print("Main container not found!")
    exit()

# Loop through items (Amazon product blocks)
for row in table.find_all('div', attrs={'class': 'a-carousel-card'}):
    quote = {}

    # Product Title
    title = row.find('span')
    quote['theme'] = title.text.strip() if title else "N/A"

    # Product URL
    link = row.find('a')
    quote['url'] = "https://www.amazon.com" + link['href'] if link else "N/A"

    # Image
    img = row.find('img')
    quote['img'] = img['src'] if img else "N/A"

    # Dummy split (since Amazon doesn't have alt like quotes)
    if img and 'alt' in img.attrs:
        parts = img['alt'].split(" ")
        quote['lines'] = parts[0] if len(parts) > 0 else "N/A"
        quote['author'] = parts[1] if len(parts) > 1 else "N/A"
    else:
        quote['lines'] = "N/A"
        quote['author'] = "N/A"

    quotes.append(quote)

# Save CSV
filename = 'amazon_data.csv'
with open(filename, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, ['theme','url','img','lines','author'])
    w.writeheader()
    for quote in quotes:
        w.writerow(quote)

print("Scraping completed!")