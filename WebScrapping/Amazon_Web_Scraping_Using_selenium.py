from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import csv

URL = "https://www.amazon.com/s?k=mobile"

options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0")

prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(30)

driver.get(URL)

wait = WebDriverWait(driver, 15)

try:
    products = wait.until(
        EC.presence_of_all_elements_located(
            (By.XPATH, "//div[@data-component-type='s-search-result']")
        )
    )
except TimeoutException:
    print("Timeout: Page blocked or too slow")
    driver.quit()
    exit()

data = []

for product in products:
    item = {}
    
    try:
        item['model'] = product.find_element(By.XPATH, ".//h2/span").text
    except:
        item['model'] = "N/A"

    try:
        item['url'] = product.find_element(By.XPATH, ".//h2/a").get_attribute("href")
    except:
        item['url'] = "N/A"

    try:
        item['img'] = product.find_element(By.XPATH, ".//img").get_attribute("src")
    except:
        item['img'] = "N/A"

    item['price'] = "N/A"
    item['dilvery_date'] = "N/A"

    data.append(item)

# Save CSV
with open("amazon_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=['model','url','img','price','dilvery_date'])
    writer.writeheader()
    writer.writerows(data)

driver.quit()
print("Scraping completed!")