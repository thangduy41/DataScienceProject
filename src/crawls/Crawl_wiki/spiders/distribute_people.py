import scrapy
import json
from collections import defaultdict

class DistributePeopleSpider(scrapy.Spider):
    name = "distributepeople"
    start_urls = [
        "https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_%C4%91%C6%A1n_v%E1%BB%8B_h%C3%A0nh_ch%C3%ADnh_c%E1%BA%A5p_huy%E1%BB%87n_c%E1%BB%A7a_Vi%E1%BB%87t_Nam"
    ]

    custom_settings = {
        'FEEDS': {}  # Không dùng FEEDS để export
    }

    def __init__(self):
        self.result = defaultdict(dict)

    def parse(self, response):
        rows = response.css('table.wikitable.sortable tr')
        for row in rows[1:]:  # Bỏ dòng tiêu đề
            cols = row.css('td')
            if len(cols) >= 9:
                ten_don_vi = cols[1].css('a::text').get('').strip().lower()
                tinh_thanh = cols[2].css('a::text').get('').strip().lower()

                dan_so = cols[5].css('::text').get('').strip()
                dien_tich = cols[6].css('::text').get('').strip()
                mat_do = cols[7].css('::text').get('').strip()
                so_xa = cols[8].css('::text').get('').strip()

                self.result[tinh_thanh][ten_don_vi] = {
                    "number_people": dan_so,
                    "area": dien_tich,
                    "distribute": mat_do,
                    "communes": so_xa
                }

    def closed(self, reason):
        with open("output/distributepeople.json", "w", encoding="utf-8") as f:
            json.dump(self.result, f, ensure_ascii=False, indent=4)
