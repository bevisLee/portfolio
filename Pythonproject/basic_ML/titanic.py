
# data를 다운로드 받자.


import wget
url = [
    "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/gender_submission.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510903672&Signature=ksSfeyLgZk3iheQrT%2F9RMXPyDyMOA5fr%2BizMBe1ARe8M2761t9NLvBIc0HFusqb17oYA%2FAGl%2BYRYhJNt0zDsEX4cNwvYs5yhB98x0xBlHoV5Xwi%2BR4D6kLXPCbKmsHXrxP8BViPM5ZWmQCu1fXjmyABoPdCbYbimGWdeu7bXNhsHFV3vaRYOEuSpK7pjNA3i7jqSlfnp8d741avzgpz88QInWV2VARi1BgvDFidWeX3ypwgfCISwqII7WpOC6fg%2BLLWDfQ7Paq0%2BoCNmRO0hcLHTggS8bec1jJ4wzEbvJxALJQsuI86MUOMs7k%2BawR8MBxbSLWz4A%2FEDGOTo6FIPrw%3D%3D",
    "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510903879&Signature=G2HzwOLDbl0NJjutuY3pTQubJFPBL2qsvA3LaLOoBd88l%2BtXwpkn%2FghcHnguW06g6Tvsx8ZxeNX20Qn6aUe7jcCeSkcfpEdX6q4wNYwdSEy4%2BGgZS49Pyuh6aU4QkocViHAbJ5BvbTfBCFGeHdU%2B%2Fu9gNXYN657IO04Wci2wGlKebWtVyDlpHOOs4NGhW5RKhyakbItpBKHkF5gCJgkTEpEtsBsUAAy%2FeVn58nnLMtvnuyNzn%2BcP4PKgF2AnygAwBxWYDTFfhMNQoh2XsK77kJVwtgRR4Egn7f5laCrREq5P38uxw%2F2LXVPWBfNWh9q86KXw0gjXmLiRQwbqNXxQow%3D%3D",
    "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510903943&Signature=WhUSOK9MVjfRwuUqf0EWnWIF7KEMLr1WRTl6zUNS3L0ZzHWqcNZCFN%2F%2BqcxA%2F9oEwt6lXiIryoo5GEixiAgz5dSx3PKsFvk0lPKTechDpiWv%2BI5w7JmJin1Sz%2F6M9%2BZasCVGLREMUtd6sHaqYSkp%2BoIkZygV8z%2BjaQJxpVoX0Lkl5531Wr%2B4L9KPJ5YOtg3MfRBFS3eBShHj3I%2F%2F2JkEXJTd9hY5ootBh%2FZ8RXPf9kr61IL93hzn9WpuavjETTc2eboQE8zp58aYYwxKAOw3h%2BQWNrU0QT2%2F%2Bs%2FXk9GZt72plme1IU1IL9ADVb9pEIn2GIaU0x9lWTbkJXsf2G3G4g%3D%3D"
      ]

for i in url:
    wget.download(i)

import os

for subdir, dirs, files in os.walk('data'):
    for file in files:
        if '.csv' in file:
            print (file)

