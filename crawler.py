import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException
)
from bs4 import BeautifulSoup


def run_crawler(max_pages: int = 10, output_dir: str = "/home/hail/RAG/data/"):
    """
    Naver Economy News Crawler (stable version + all categories)
    - max_pages: number of times to click 'more' button
    - output_dir: directory to save CSV files
    """

    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://news.naver.com/")
    time.sleep(2)

    # Navigate to the Economy section
    economy_btn = driver.find_element(
        By.CSS_SELECTOR,
        "body > section > header > div.Nlnb._float_lnb > div > div > div > div > div > ul > li:nth-child(3) > a"
    )
    economy_btn.click()
    time.sleep(2)

    # Economy categories
    categories = {
        "finance": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(1)",
        "securities": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(2)",
        "industry": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(3)",
        "venture": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(4)",
        "realestate": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(5)",
        "global": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(6)",
        "life": "#ct_wrap > div.ct_scroll_wrapper > div.column0 > div > ul > li:nth-child(7)",
    }

    for cat, selector in categories.items():
        print(f"Start category: {cat}")
        driver.find_element(By.CSS_SELECTOR, selector).click()
        time.sleep(2)

        articles = []

        for page in range(max_pages):
            try:
                # Wait until article titles are loaded in DOM
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.sa_text_title"))
                )
                titles = driver.find_elements(By.CSS_SELECTOR, "a.sa_text_title")
            except TimeoutException:
                print("Timeout: Failed to load article titles")
                break

            # Process only new articles (skip already collected)
            for idx in range(len(articles), len(titles)):
                retry = 0
                while retry < 3:
                    try:
                        # Re-fetch elements every loop to avoid stale reference
                        titles = driver.find_elements(By.CSS_SELECTOR, "a.sa_text_title")
                        t = titles[idx]

                        title = t.text.strip()
                        link = t.get_attribute("href")

                        # Open article in a new tab
                        driver.execute_script("window.open(arguments[0]);", link)
                        driver.switch_to.window(driver.window_handles[-1])
                        time.sleep(1)

                        # Parse article content
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
                        break
                    except StaleElementReferenceException:
                        retry += 1
                        time.sleep(0.5)
                        if retry == 3:
                            print("Skip one article due to stale element")

            # Try to click 'more' button for next page
            try:
                more_btn = driver.find_element(
                    By.CSS_SELECTOR, "#newsct > div.section_latest > div > div.section_more > a"
                )
                driver.execute_script("arguments[0].click();", more_btn)
                time.sleep(2)
            except Exception:
                print(f"No more pages in {cat}")
                break

        # Save results as CSV
        df = pd.DataFrame(articles)
        output_path = f"{output_dir}/naver_news_{cat}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {cat}: {len(articles)} articles")

        # Scroll back to top before moving to next category
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1.5)

    driver.quit()

if __name__ == "__main__":
    run_crawler(max_pages=10)
