from selenium import webdriver
from selenium.webdriver.common import by


class TickerRobber:
    def __init__(self, url, username, password, xpath=None):
        self.url = url
        self.driver = webdriver.Chrome(keep_alive=True)
        self.driver.get(url)
        self.username = username
        self.password = password
        self.xpath = xpath

    def login(self):
        self.driver.find_element(value='username').send_keys('your_username')
        self.driver.find_element(value='password').send_keys('your_password')
        self.cookies = self.driver.get_cookies()

    def apply(self, xpath=None):
        driver.find_element(by=by.By.XPATH, value=self.xpath).click()
        driver.find_element(value='seat_id').click()
        driver.find_element(value='confirm_button').click()
        driver.find_element(value='payment').click()
        driver.find_element(value='password').send_keys('your_password')

        driver.find_element(value='submit_button').click()

        driver.find_element(by=by.By.XPATH, value='success_xpath')


if __name__ == "__main__":
    robber = TickerRobber(url='https://www.gewara.com/detail/290317', username="15088363086", password="mario19,.")
