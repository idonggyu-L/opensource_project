from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import pandas as pd
import time

driver = webdriver.Chrome()
driver.maximize_window()

driver.get("https://news.naver.com/")
time.sleep(2)

# Click on Economy menu button
economy_btn = driver.find_element(
    By.CSS_SELECTOR,
    "body > section > header > div.Nlnb._float_lnb > div > div > div > div > div > ul > li:nth-child(3) > a"
)
economy_btn.click()
time.sleep(2)

categories = {
    "finance": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(1)" #,
    #"securities": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(2)",
    #"industry": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(3)",
    #"venture": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(4)",
    #"realestate": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(5)",
    #"global": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(6)",
    #"life": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(7)"
}

all_results = {}

for cat, selector in categories.items():
    print(f"Start category: {cat}")
    driver.find_element(By.CSS_SELECTOR, selector).click()
    time.sleep(2)

    articles = []
    prev_count = 0

    i = 0

    while i <= 100:
        titles = driver.find_elements(By.CSS_SELECTOR, "a.sa_text_title")

        for t in titles[len(articles):]:
            title = t.text.strip()
            link = t.get_attribute("href")
            driver.execute_script("window.open(arguments[0]);", link)
            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(1.5)

            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(1)

            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(1)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            content_tag = soup.select_one("article#dic_area")
            content = content_tag.get_text(" ", strip=True) if content_tag else ""

            articles.append({
                "title": title,
                "link": link,
                "content": content
            })

            driver.close()
            driver.switch_to.window(driver.window_handles[0])


        # Scroll down to load new articles
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        more_btn = driver.find_element(By.CSS_SELECTOR, "#newsct > div.section_latest > div > div.section_more > a ")
        driver.execute_script("arguments[0].click();", more_btn)
        time.sleep(2)

    # Save results
    all_results[cat] = pd.DataFrame(articles)
    all_results[cat].to_csv(f"naver_news_{cat}.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {cat}: {len(articles)} articles")
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1.5)

driver.quit()
