import scrapy


class DistributepeopleSpider(scrapy.Spider):
    name = "distributepeople"
    allowed_domains = ["vi.wikipedia.org"]
    start_urls = ["https://vi.wikipedia.org"]

    def parse(self, response):
        pass
