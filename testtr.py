import os 
# 修改package下载的目录
os.environ['XDG_DATA_HOME']='./.local/share'
os.environ['XDG_CACHE_HOME']='./.local/cache'
import argostranslate.translate
import argostranslate.package

# 下载所有的包
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
for available_package in available_packages:
    argostranslate.package.install_from_path(available_package.download())

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='Translate')

class Translaterequest(BaseModel):
    query: str
    from_code: str
    to_code: str

@app.post('/translate')
async def api_translate(translaterequest: Translaterequest):
    try:
        translatedText = argostranslate.translate.translate(translaterequest.query,translaterequest.from_code,translaterequest.to_code)
    except Exception as e:
        ret = {'results': e, 'code': 500}
        return ret
    ret = {'results': translatedText, 'code': 0}
    return ret

if __name__ == '__main__':
    uvicorn.run('example-fastapi:app',host='0.0.0.0',workers=1)