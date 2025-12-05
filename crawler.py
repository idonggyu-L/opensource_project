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


def run_crawler(max_pages: int = 20, output_dir: str = "/home/hail/Desktop/RAG_clean/data/"):
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


        # for page in range(max_pages):
        #     print(f"[DEBUG] Page={page + 1}")
        #
        #     WebDriverWait(driver, 10).until(
        #         EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.sa_text_title"))
        #     )
        #     if page==0:
        #         titles= driver.find_elements(By.CSS_SELECTOR, "a.sa_text_title")
        #
        #     else:
        #         _new = driver.find_elements(By.CSS_SELECTOR, "a.sa_text_title")
        #         titles = list(set(_new+titles))
        #
        #     for idx, t in enumerate(titles):
        #         try:
        #             title = t.text.strip()
        #             link = t.get_attribute("href")
        #
        #             driver.execute_script("window.open(arguments[0]);", link)
        #             driver.switch_to.window(driver.window_handles[-1])
        #             time.sleep(1)
        #
        #             soup = BeautifulSoup(driver.page_source, "html.parser")
        #             content_tag = soup.select_one("article#dic_area")
        #             content = content_tag.get_text(" ", strip=True) if content_tag else ""
        #
        #             articles.append({"title": title, "link": link, "content": content})
        #
        #             driver.close()
        #             driver.switch_to.window(driver.window_handles[0])
        #
        #         except Exception as e:
        #             print(f"[ERROR] Content parsing failed: {e}")
        #             continue
        #
        #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        #     time.sleep(2)
        #
        #
        #     more_btn = driver.find_element(
        #         By.CSS_SELECTOR, "#newsct > div.section_latest > div > div.section_more > a"
        #     )
        #     driver.execute_script("arguments[0].scrollIntoView(true);", more_btn)
        #     time.sleep(2)
        #     driver.execute_script("arguments[0].click();", more_btn)
        #     time.sleep(2)
        #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Prevent duplicate articles using a set of URLs
        seen_links = set()

        for page in range(max_pages):
            print(f"[DEBUG] PAGE {page + 1}")

            # Wait until article titles are present
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.sa_text_title"))
            )

            # Collect the current page titles
            articles_el = driver.find_elements(By.CSS_SELECTOR, "a.sa_text_title")

            # Parse only new articles (avoid duplicates using href)
            for t in articles_el:
                link = t.get_attribute("href")

                # Skip duplicated articles
                if link in seen_links:
                    continue
                seen_links.add(link)

                title = t.text.strip()

                # Open and parse the article
                driver.execute_script("window.open(arguments[0]);", link)
                driver.switch_to.window(driver.window_handles[-1])
                time.sleep(1)

                soup = BeautifulSoup(driver.page_source, "html.parser")
                content_tag = soup.select_one("article#dic_area")
                content = content_tag.get_text(" ", strip=True) if content_tag else ""

                driver.close()
                driver.switch_to.window(driver.window_handles[0])

                articles.append({"title": title, "link": link, "content": content})

            # Scroll down to expose 'More' button (required for AJAX loading)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # Try to click the 'More' button to load next page
            try:
                more_btn = driver.find_element(
                    By.CSS_SELECTOR,
                    "#newsct > div.section_latest > div > div.section_more > a"
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", more_btn)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", more_btn)
                time.sleep(2)

            except Exception:
                print("No more pages available")
                break

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
