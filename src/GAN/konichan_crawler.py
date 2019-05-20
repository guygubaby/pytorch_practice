import requests
import os
from bs4 import BeautifulSoup
import traceback
import multiprocessing as mp


class Konichan:
    url_template='http://konachan.net/post?page={}&tags='

    def __init__(self,start=1,end=8):
        self.start=start
        self.end=end

        if os.path.exists('imgs') is False:
            os.makedirs('imgs')

    def init_urls(self):
        return [self.url_template.format(i) for i in range(self.start,self.end)]

    def download(self,url,filename):
        if os.path.exists(filename):
            print('file exists!')
            return
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
            return filename
        except KeyboardInterrupt:
            if os.path.exists(filename):
                os.remove(filename)
            raise KeyboardInterrupt
        except Exception:
            traceback.print_exc()
            if os.path.exists(filename):
                os.remove(filename)

    def start_crawl(self):
        for url in self.init_urls():
            res=requests.get(url).text
            soup=BeautifulSoup(res,'html.parser')
            for img in soup.find_all('img', class_="preview"):
                target_url =  img['src']
                filename = os.path.join('imgs', target_url.split('/')[-1])
                self.download(target_url,filename)
            print(f'page {url} end')


if __name__ == '__main__':
    konichang=Konichan(start=1,end=20)
    # konichang.start_crawl()

    pool_size=10
    pool=mp.Pool(pool_size)
    pool.apply_async(konichang.start_crawl())

    pool.close()
    pool.join()
    print('ok :)')
