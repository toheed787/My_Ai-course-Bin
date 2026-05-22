from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import csv

URL = "https://www.daraz.pk/catalog/?q=smartphones"

options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)

driver.get(URL)
time.sleep(10)  # VERY IMPORTANT for Daraz

# get all cards
cards = driver.find_elements(By.CSS_SELECTOR, "div[data-qa-locator='product-item']")

print(f"Found {len(cards)} product cards")

data = []

for i, card in enumerate(cards):
    print(f"Processing product {i+1}...")

    try:
        # FULL TEXT (IMPORTANT FIX)
        text = card.text.split("\n")

        if len(text) < 2:
            continue

        title = text[0]
        price = "N/A"

        # find price inside text list
        for t in text:
            if "Rs" in t or "PKR" in t:
                price = t
                break

        # URL
        try:
            url = card.find_element(By.TAG_NAME, "a").get_attribute("href")
        except:
            url = "N/A"

        # IMAGE
        try:
            img = card.find_element(By.TAG_NAME, "img").get_attribute("src")
        except:
            img = "N/A"

        if title.strip() != "":
            data.append({
                "title": title,
                "price": price,
                "url": url,
                "img": img
            })

    except:
        continue

print(f"\nSuccessfully scraped {len(data)} products")

# SAVE CSV
with open("daraz_final.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title","price","url","img"])
    writer.writeheader()
    writer.writerows(data)

driver.quit()

print("Scraping completed!")