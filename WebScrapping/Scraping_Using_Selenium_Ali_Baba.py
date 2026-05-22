# Python program to scrape Alibaba Search Results using Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv
import random
from webdriver_manager.chrome import ChromeDriverManager

# URL for Alibaba Auto Accessories search
URL = "https://www.alibaba.com/trade/search?SearchText=Auto+Accessories&indexArea=product_en&tab=all"

# Configure Chrome options to avoid detection
chrome_options = Options()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
chrome_options.add_argument("--lang=en-US")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

CHROME_DRIVER_PATH = r"C:\Users\User\Downloads\chromedriver-win64.zip"  # <--- UPDATE THIS PATH

service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install())
)
# Anti-detection measures
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
    'source': '''
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        })
    '''
})

driver.get(URL)

# Wait for page to load
wait = WebDriverWait(driver, 20)
time.sleep(random.uniform(5, 8))  # Initial wait

products = []

# ========== STEP 1: FIND PRODUCT CONTAINERS ==========
product_cards = []
selectors_to_try = [
    "div[data-role='offer-item']",      # Common Alibaba offer item
    "div[class*='offer-item']",
    "div[class*='product-item']",
    "div[class*='search-item']",
    "div[data-content='product-item']",
    "div[class*='J-offer-item']"
]

print("Looking for product containers...")
for selector in selectors_to_try:
    try:
        cards = driver.find_elements(By.CSS_SELECTOR, selector)
        if cards:
            product_cards = cards
            print(f"Found {len(product_cards)} products using selector: {selector}")
            break
    except:
        continue

# Fallback: Wait for any element that might contain products
if not product_cards:
    print("Trying XPath fallback...")
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'offer')]")))
        product_cards = driver.find_elements(By.XPATH, "//div[contains(@class, 'offer')]")
        print(f"Found {len(product_cards)} products using XPath fallback")
    except TimeoutException:
        print("Timeout: No products found with standard selectors.")

# Save debug page source for inspection
if not product_cards:
    with open('alibaba_debug.html', 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print("No products found! Saved debug HTML to 'alibaba_debug.html'")
    print("Please open this file to inspect the actual class names.")
    driver.quit()
    exit()

# ========== STEP 2: EXTRACT PRODUCT DATA ==========
for idx, card in enumerate(product_cards[:30], 1):
    product = {}
    print(f"\nProcessing product {idx}...")

    # 2.1 Product Title
    title = None
    title_selectors = [
        ".//div[contains(@class, 'title')]//a",
        ".//a[contains(@class, 'title')]",
        ".//p[contains(@class, 'title')]",
        ".//h2[contains(@class, 'title')]",
        ".//a[contains(@href, 'product-detail')]"
    ]
    for selector in title_selectors:
        try:
            elem = card.find_element(By.XPATH, selector)
            title = elem.text.strip()
            if title:
                break
        except:
            continue
    product['title'] = title if title else "N/A"
    print(f"  Title: {product['title'][:60] if product['title'] != 'N/A' else 'N/A'}")

    # 2.2 Product URL
    url = None
    for selector in title_selectors:
        try:
            elem = card.find_element(By.XPATH, selector)
            href = elem.get_attribute("href")
            if href:
                url = href if href.startswith('http') else f"https:{href}"
                break
        except:
            continue
    product['url'] = url if url else "N/A"

    # 2.3 Price
    price = None
    price_selectors = [
        ".//span[contains(@class, 'price')]",
        ".//div[contains(@class, 'price')]",
        ".//span[contains(@class, 'value')]",
        ".//div[@data-role='price']"
    ]
    for selector in price_selectors:
        try:
            elem = card.find_element(By.XPATH, selector)
            price = elem.text.strip()
            if price:
                break
        except:
            continue
    product['price'] = price if price else "N/A"
    print(f"  Price: {product['price']}")

    # 2.4 Supplier/Company Name
    supplier = None
    supplier_selectors = [
        ".//a[contains(@class, 'supplier')]",
        ".//div[contains(@class, 'company')]",
        ".//div[contains(@class, 'supplier-name')]"
    ]
    for selector in supplier_selectors:
        try:
            elem = card.find_element(By.XPATH, selector)
            supplier = elem.text.strip()
            if supplier:
                break
        except:
            continue
    product['supplier'] = supplier if supplier else "N/A"

    # 2.5 Rating
    rating = None
    rating_selectors = [
        ".//div[contains(@class, 'rating')]",
        ".//span[contains(@class, 'star')]",
        ".//div[@data-role='rating']"
    ]
    for selector in rating_selectors:
        try:
            elem = card.find_element(By.XPATH, selector)
            rating = elem.text.strip()
            if rating:
                break
        except:
            continue
    product['rating'] = rating if rating else "N/A"

    # 2.6 Orders / Reviews
    orders = None
    orders_selectors = [
        ".//span[contains(@class, 'order')]",
        ".//div[@data-role='order-count']",
        ".//span[contains(text(), 'Order')]"
    ]
    for selector in orders_selectors:
        try:
            elem = card.find_element(By.XPATH, selector)
            orders = elem.text.strip()
            if orders:
                break
        except:
            continue
    product['orders'] = orders if orders else "N/A"

    # 2.7 Image URL
    img_url = None
    try:
        img = card.find_element(By.XPATH, ".//img[contains(@class, 'img')]")
        img_url = img.get_attribute("src") or img.get_attribute("data-src")
    except:
        pass
    product['image_url'] = img_url if img_url else "N/A"

    # Add only if we have a title
    if product['title'] != "N/A":
        products.append(product)

# ========== STEP 3: SAVE RESULTS ==========
print(f"\n{'='*60}")
print(f"Successfully scraped {len(products)} products")
print(f"{'='*60}")

if products:
    filename = 'alibaba_products.csv'
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['title', 'price', 'supplier', 'rating', 'orders', 'url', 'image_url']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for product in products:
            writer.writerow(product)
    print(f"✓ Data saved to '{filename}'")

    print("\nSample of scraped data (first 3):")
    for i, p in enumerate(products[:3], 1):
        print(f"\nProduct {i}:")
        print(f"  Title: {p['title'][:80]}...")
        print(f"  Price: {p['price']}")
        print(f"  Supplier: {p['supplier']}")
        print(f"  Rating: {p['rating']}")
        print(f"  Orders: {p['orders']}")
else:
    print("⚠️ No products were scraped.")
    print("\nPlease inspect 'alibaba_debug.html' to identify the correct CSS selectors.")

driver.quit()
print("\nScraping completed!")