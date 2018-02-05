
# web page - https://www.liip.ch/en/blog/exploring-google-analytics-data-with-python-pandas

import pandas
import google2pandas

from google2pandas import *
conn = GoogleAnalyticsQuery(secrets='C:/Users/bevis/Downloads/bevis-sandbox-afaa135a8c48.json') #, token_file_name='./ga-creds/analytics.dat')

