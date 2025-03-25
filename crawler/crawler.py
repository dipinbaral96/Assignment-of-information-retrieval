import time
import json
import schedule
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting"

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

def crawl_data():
    driver = setup_driver()
    
    # Store results
    data = {
        "publications": [],
        "authors": [],
        "year_trends": {}
    }
    
    # Crawl Publications & Extract Authors
    page = 0
    while True:
        url = f"{BASE_URL}/publications/?page={page}"
        print(f"Fetching: {url}")
        driver.get(url)
        time.sleep(5)
        
        items = driver.find_elements(By.CSS_SELECTOR, "div.result-container")
        
        if not items:
            break

        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, "h3.title").text
                link = item.find_element(By.CSS_SELECTOR, "h3.title > a").get_attribute('href')
                year = item.find_element(By.CSS_SELECTOR, "span.date").text
                
                author_elements = item.find_elements(By.CSS_SELECTOR, "a[rel='Person']")
                authors = []

                for author in author_elements:
                    author_name = author.text.strip()
                    author_link = author.get_attribute('href')

                    if author_name and author_link:
                        authors.append({"name": author_name, "profile_link": author_link})

                        # Store authors separately (avoid duplicates)
                        if author_name not in [a["name"] for a in data["authors"]]:
                            data["authors"].append({"name": author_name, "profile_link": author_link})
                
                data["publications"].append({
                    "title": title,
                    "link": link,
                    "authors": authors,
                    "year": year
                })
            
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        page += 1

    # Crawl Year Trends
    driver.get(BASE_URL)
    time.sleep(5)
    
    try:
        start_year = int(driver.find_element(By.CSS_SELECTOR, "span.year-start").text)
        end_year = int(driver.find_element(By.CSS_SELECTOR, "span.year-end").text)
        bars = driver.find_elements(By.CSS_SELECTOR, "span.bar")
        
        for i, year in enumerate(range(start_year, end_year + 1)):
            if 2019 <= year <= 2025:
                data["year_trends"][year] = bars[i].get_attribute("style")
    except:
        pass

    driver.quit()

    # Save to JSON
    with open("./files/res.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("Data saved to res.json")

# Schedule the crawler
schedule.every(1).minutes.do(crawl_data)

if __name__ == "__main__":
    crawl_data()
    while True:
        schedule.run_pending()
        time.sleep(1)
