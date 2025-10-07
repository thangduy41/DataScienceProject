# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


# class PredihomeItem(scrapy.Item):
#     # define the fields for your item here like:
#     # name = scrapy.Field()
#     pass

class PredihomeItem(scrapy.Item):
    stt = scrapy.Field()
    ten_don_vi = scrapy.Field()
    tinh_thanh = scrapy.Field()
    dan_so = scrapy.Field()
    dien_tich = scrapy.Field()
    mat_do = scrapy.Field()
    so_xa = scrapy.Field()